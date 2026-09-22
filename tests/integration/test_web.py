import http.client
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from wf_forward import ConfigurationError, DeltaInitialCondition, ForwardSolver, SolverConfig, UniformGrid, WrightFisherModel
from wf_forward.web import JobManager, WebServer, density_frame, validate_web_config


def config():
    return {"schema_version": 1, "model": {"Ne": 100}, "grid": {"points": 101},
            "initial_condition": {"type": "delta", "x0": .3},
            "solver": {"output_times": [0, 1, 2]}}


@pytest.mark.parametrize("change", [
    {"grid": {"points": 100_000_001}}, {"grid": {"points": True}},
    {"solver": {"output_times": list(range(202))}}, {"solver": {"output_times": "bad"}},
    {"solver": {"max_steps": 200_001}}, {"solver": {"max_output_bytes": 10**12}},
    {"output": {"summary_csv": "/tmp/unrequested.csv"}}, {"model": {"Ne": -1}},
])
def test_web_validation_limits(change):
    with pytest.raises(ConfigurationError):
        validate_web_config(config() | change)


def test_density_aggregation_preserves_mass():
    result = ForwardSolver(WrightFisherModel(Ne=1000), UniformGrid(2001),
                           SolverConfig(output_times=[0, .01])).solve(DeltaInitialCondition(.3002))
    for index in (0, 1):
        frame = density_frame(result, index, max_bins=101)
        assert len(frame["x"]) == 101 and frame["aggregated"]
        assert np.dot(frame["density"], frame["widths"]) == pytest.approx(result.p_segregating[index], abs=1e-14)


@pytest.fixture
def server():
    app = WebServer(0)
    thread = threading.Thread(target=app.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    yield app
    app.shutdown()
    thread.join(timeout=3)
    app.server_close()


def request(server, path, data=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    kwargs = dict(headers or {})
    if data is not None or raw is not None:
        kwargs.setdefault("Content-Type", "application/json")
        kwargs.setdefault("X-WF-Token", server.token)
        body = raw if raw is not None else json.dumps(data)
        conn.request("POST", path, body=body, headers=kwargs)
    else:
        conn.request("GET", path, headers=kwargs)
    response = conn.getresponse()
    code, response_headers, body = response.status, dict(response.getheaders()), response.read()
    conn.close()
    return code, response_headers, body


def await_job(server, job_id):
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        _, _, body = request(server, f"/api/jobs/{job_id}")
        info = json.loads(body)
        if info["status"] != "running":
            return info
        time.sleep(.02)
    raise AssertionError("worker did not finish")


def test_web_assets_and_request_boundaries(server):
    for path in ("/", "/style.css", "/app.js"):
        code, headers, body = request(server, path)
        assert code == 200 and len(body) > 100
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert request(server, "/../pyproject.toml")[0] == 404
    assert request(server, "/api/session", headers={"Host": "attacker.example"})[0] == 403
    assert request(server, "/api/session", headers={"Origin": "https://attacker.example"})[0] == 403
    assert request(server, "/api/jobs", data=config(), headers={"X-WF-Token": "bad"})[0] == 403
    assert request(server, "/api/jobs", data=config(), headers={"Content-Type": "text/plain"})[0] == 415
    assert request(server, "/api/jobs", raw='{"NaN": NaN}')[0] == 422
    assert request(server, "/api/jobs", raw=" "+"x"*131072)[0] == 413
    assert request(server, "/api/jobs", data=config() | {"model": {"Ne": -1}})[0] == 422


def test_web_solver_download_and_frame_match_api(server, tmp_path):
    code, _, body = request(server, "/api/jobs", data=config())
    assert code == 202
    job = json.loads(body)
    assert await_job(server, job["id"])["status"] == "complete"
    base = f"/api/jobs/{job['id']}"
    _, _, body = request(server, base+"/summary.json")
    summary = json.loads(body)
    assert summary["mean"] == pytest.approx([.3]*3, abs=1e-12)
    assert summary["metadata"]["endpoint_semantics"] == "absorption"
    assert request(server, base+"/frame?index=-1")[0] == 422
    assert request(server, base+"/frame?index=bad")[0] == 422
    code, _, frame = request(server, base+"/frame?index=2")
    assert code == 200 and json.loads(frame)["time"] == 2
    code, headers, blob = request(server, base+"/result.npz")
    assert code == 200 and "attachment" in headers["Content-Disposition"]
    path = tmp_path/"download.npz"
    path.write_bytes(blob)
    from wf_forward import SimulationResult
    downloaded = SimulationResult.load_npz(path)
    from wf_forward.cli import from_config
    solver, initial, _ = from_config(config())
    np.testing.assert_array_equal(downloaded.probability_mass, solver.solve(initial).probability_mass)
    assert b"p_loss,p_fix" in request(server, base+"/summary.csv")[2]
    normalized = json.loads(request(server, base+"/config.json")[2])
    assert normalized["solver"]["max_steps"] == 200000
    assert request(server, "/api/jobs/missing")[0] == 404


def test_web_cancel_concurrent_run_and_restart(server):
    slow = config() | {"grid": {"points": 20001}, "solver": {"output_times": [0, 10000], "method": "expm"}}
    code, _, body = request(server, "/api/jobs", data=slow)
    assert code == 202
    job_id = json.loads(body)["id"]
    assert request(server, "/api/jobs", data=config())[0] == 409
    assert request(server, f"/api/jobs/{job_id}/result.npz")[0] == 409
    code, _, body = request(server, f"/api/jobs/{job_id}/cancel", data={})
    assert code == 200 and json.loads(body)["status"] == "cancelled"
    assert server.manager.get(job_id).process.poll() is not None
    code, _, body = request(server, "/api/jobs", data=config())
    assert code == 202
    assert await_job(server, json.loads(body)["id"])["status"] == "complete"


def test_worker_error_and_timeout():
    manager = JobManager(timeout=15)
    try:
        invalid_initial = config() | {"initial_condition": {"type": "array", "values": [1, 2]}}
        job = manager.start(invalid_initial)
        manager.watchers[-1].join(timeout=15)
        assert manager.get(job["id"]).status == "failed"
        assert "mass" in manager.get(job["id"]).error
    finally:
        manager.close()
    manager = JobManager(timeout=.001)
    try:
        job = manager.start(config())
        manager.watchers[-1].join(timeout=10)
        assert manager.get(job["id"]).status == "failed"
        assert "exceeded" in manager.get(job["id"]).error
        assert manager.get(job["id"]).process.poll() is not None
    finally:
        manager.close()


def test_bind_failure_keeps_original_error(monkeypatch):
    import socket
    def fail(*args):
        raise OSError("port occupied")
    monkeypatch.setattr(socket.socket, "bind", fail)
    with pytest.raises(OSError, match="port occupied"):
        WebServer(8765)


def await_sample(server, base):
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        info = json.loads(request(server, base+'/sampling')[2])
        if info['current']['status'] != 'running':
            return info
        time.sleep(.02)
    raise AssertionError('sampling did not finish')


def test_web_sampling_and_downloads(server, tmp_path):
    from wf_forward import SimulationResult
    job = json.loads(request(server, '/api/jobs', data=config())[2])
    assert await_job(server, job['id'])['status'] == 'complete'
    base = f"/api/jobs/{job['id']}"
    assert request(server, base+'/trajectories.json')[0] == 409
    for bad in ({'paths':129}, {'seed':-1}, {'paths':True}, {'seed':2**32}, {'unknown':1}, []):
        assert request(server, base+'/sample', data=bad)[0] == 422
    assert request(server, base+'/sample', data={'paths':12, 'seed':9})[0] == 202
    assert request(server, '/api/jobs', data=config())[0] == 409
    assert request(server, base+'/sample', data={})[0] == 409
    report = await_sample(server, base)
    assert report['current']['status'] == 'complete' and report['available']
    sampled = json.loads(request(server, base+'/trajectories.json')[2])
    source = tmp_path/'source.npz'
    source.write_bytes(request(server, base+'/result.npz')[2])
    expected = SimulationResult.load_npz(source).sample_trajectories(paths=12, seed=9)
    np.testing.assert_array_equal(sampled['frequencies'], expected.frequencies)
    for ext in ('csv', 'npz'):
        code, headers, data = request(server, base+'/trajectories.'+ext)
        assert code == 200 and 'attachment' in headers['Content-Disposition']
        assert len(data) > 100
    # A cancellation retains the last completed sample and releases the worker slot.
    assert request(server, base+'/sample', data={'paths':128})[0] == 202
    response = request(server, base+'/cancel-sampling', data={})
    assert json.loads(response[2])['status'] == 'cancelled'
    assert json.loads(request(server, base+'/trajectories.json')[2]) == sampled
    # A timed-out replacement likewise must not publish partial output.
    server.manager.timeout = .001
    assert request(server, base+'/sample', data={})[0] == 202
    report = await_sample(server, base)
    assert report['current']['status'] == 'failed' and report['available']
    assert json.loads(request(server, base+'/trajectories.json')[2]) == sampled


def test_sfs_jobs_exports_and_separate_session(server, tmp_path):
    from wf_forward import calculate_sfs
    config_sfs={'n':20,'theta':2,'history':{'times':[0,500], 'sizes':[10000,20000]},'output_times':[0,500,1000]}
    simulation=json.loads(request(server,'/api/jobs',data=config())[2])
    assert await_job(server,simulation['id'])['status']=='complete'
    code,_,body=request(server,'/api/sfs',data=config_sfs)
    assert code==202
    job=json.loads(body);base=f"/api/jobs/{job['id']}"
    assert job['kind']=='sfs'
    assert request(server,'/api/jobs',data=config())[0]==409
    assert await_job(server,job['id'])['status']=='complete'
    result=json.loads(request(server,base+'/sfs.json')[2])
    np.testing.assert_allclose(result['derived'],calculate_sfs(config_sfs).derived)
    assert request(server,base+'/frame')[0]==404
    assert request(server,base+'/sample',data={})[0]==422
    for ext in ('csv','npz'):
        code,headers,body=request(server,base+'/sfs.'+ext)
        assert code==200 and 'attachment' in headers['Content-Disposition']
        (tmp_path/('sfs.'+ext)).write_bytes(body)
    with np.load(tmp_path/'sfs.npz',allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['symmetric'],result['symmetric'])
    session=json.loads(request(server,'/api/session')[2])
    assert session['latest']['id']==simulation['id']
    assert session['latest_sfs']['id']==job['id']
    assert request(server,'/api/sfs',data={'n':5001})[0]==422
    assert request(server,'/api/sfs',data={},headers={'X-WF-Token':'bad'})[0]==403


def test_sfs_cancel_and_retention_per_kind(server):
    server.manager.keep=1
    main=json.loads(request(server,'/api/jobs',data=config())[2])
    assert await_job(server,main['id'])['status']=='complete'
    first=json.loads(request(server,'/api/sfs',data={'n':10})[2])
    assert await_job(server,first['id'])['status']=='complete'
    second=json.loads(request(server,'/api/sfs',data={'n':5000,
        'history':{'times':[0,10000], 'sizes':[10000,100]},'output_times':[0,10000]})[2])
    assert request(server,f"/api/jobs/{second['id']}/cancel",data={})[0]==200
    assert server.manager.get(second['id']).status=='cancelled'
    assert request(server,f"/api/jobs/{main['id']}/summary.json")[0]==200
    assert request(server,f"/api/jobs/{first['id']}")[0]==404
