"""Compile the real worker against a fake LimeSuite; never link/open hardware."""
from pathlib import Path
import json
import os
import runpy
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
HEADERS = REPO.parent/"results/calibration_progress_2026-09-09/sdr/smoke-build/include"
DRIVER = runpy.run_path(str(REPO/"tools/lime_smoke_capture_v1.py"))
FAKE = r"""
#include <lime/LimeSuite.h>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
static double rate=2e6, freq[2]={}, bw[2]={};
static unsigned gain[2]={}; static int antenna[2]={};
static std::atomic<unsigned long long> samples{0};
static bool mode(const char* s) { const char* p=getenv("FAKE_MODE"); return p && std::string(p)==s; }
extern "C" {
const char* LMS_GetLibraryVersion(){return "23.11.0-FAKE-OFFLINE-TEST";}
const char* LMS_GetLastErrorMessage(){return "fake injected failure";}
int LMS_GetDeviceList(lms_info_str_t* d){if(d)std::strcpy(d[0],"LimeSDR-Mini, serial=00001d423d9108f273");return 1;}
int LMS_Open(lms_device_t** d,const lms_info_str_t,void*){*d=reinterpret_cast<void*>(1);return 0;}
int LMS_Init(lms_device_t*){return 0;} int LMS_Close(lms_device_t*){return 0;}
int LMS_EnableChannel(lms_device_t*,bool,size_t,bool){return 0;}
int LMS_SetSampleRate(lms_device_t*,double v,size_t){rate=v;return 0;}
int LMS_GetSampleRate(lms_device_t*,bool,size_t,double* h,double* r){*h=rate;*r=rate;return 0;}
int LMS_SetLOFrequency(lms_device_t*,bool d,size_t,double v){freq[d]=v;return 0;}
int LMS_GetLOFrequency(lms_device_t*,bool d,size_t,double* v){*v=freq[d];return 0;}
int LMS_SetLPFBW(lms_device_t*,bool d,size_t,double v){bw[d]=v;return 0;}
int LMS_GetLPFBW(lms_device_t*,bool d,size_t,double* v){*v=bw[d];return 0;}
int LMS_SetGaindB(lms_device_t*,bool d,size_t,unsigned v){gain[d]=v;return 0;}
int LMS_GetGaindB(lms_device_t*,bool d,size_t,unsigned* v){*v=gain[d];return 0;}
int LMS_SetAntenna(lms_device_t*,bool d,size_t,size_t v){
 if(d && v==LMS_PATH_TX2){
  if(mode("antenna_fail") || samples<static_cast<unsigned long long>(rate*.1))return -1;
  const char* marker=getenv("FAKE_TX_MARKER");if(marker)std::ofstream(marker)<<"TX after RX ready";
 }
 antenna[d]=v;return 0;
}
int LMS_GetAntenna(lms_device_t*,bool d,size_t){return antenna[d];}
int LMS_SetupStream(lms_device_t*,lms_stream_t*){return 0;}
int LMS_DestroyStream(lms_device_t*,lms_stream_t*){return 0;}
int LMS_StartStream(lms_stream_t*){return 0;}
int LMS_StopStream(lms_stream_t*){return 0;}
int LMS_RecvStream(lms_stream_t*,void* output,size_t n,lms_stream_meta_t* m,unsigned){
 std::this_thread::sleep_for(std::chrono::milliseconds(mode("slow")?10:1));
 if(mode("no_rx"))return -1;
 m->timestamp=samples.load(); if(mode("gap") && samples>0)m->timestamp+=1;
 auto* p=static_cast<std::complex<float>*>(output);
 for(size_t i=0;i<n;++i)p[i]=mode("clip")?std::complex<float>(1,0):std::complex<float>(.001F,.002F);
 samples+=n;return static_cast<int>(n);
}
int LMS_SendStream(lms_stream_t*,const void*,size_t n,const lms_stream_meta_t* m,unsigned){
 if(mode("tx_fail") || samples<static_cast<unsigned long long>(rate*.1) || !m->waitForTimestamp)return -1;
 return static_cast<int>(n);
}
int LMS_GetStreamStatus(lms_stream_t*,lms_stream_status_t* s){std::memset(s,0,sizeof(*s));return 0;}
}
"""


@pytest.fixture(scope="session")
def fake_build(tmp_path_factory):
    if not (HEADERS/"lime/LimeSuite.h").exists():
        pytest.skip("preserved LimeSuite 23.11 headers are required for C++ offline tests")
    base = tmp_path_factory.mktemp("fake-lime-smoke-only")
    source = base/"fake.cpp"
    source.write_text(FAKE)
    library = base/"libFakeLimeSmoke.so"
    worker = base/"worker"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-pthread", "-I", str(HEADERS), str(source), "-o", str(library)], check=True)
    subprocess.run(["g++", "-std=c++17", "-pthread", "-I", str(HEADERS), str(REPO/"tools/lime_smoke_worker_v1.cpp"), str(library), "-o", str(worker)], check=True)
    manifest = base/"build.json"
    manifest.write_text(json.dumps({"worker": str(worker), "worker_sha256": DRIVER["sha"](worker),
                                   "library": str(library), "inputs": {str(source): DRIVER["sha"](source), str(library): DRIVER["sha"](library)}}))
    return manifest


def prepared(tmp_path, fake_build):
    output = tmp_path/"attempt"
    DRIVER["prepare"](output, build_manifest=fake_build, serial="0x1d423d9108f273", frequency_hz=500e6)
    return output


def test_preparation_never_launches_and_requires_new_directory(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    tx = np.fromfile(out/"tx.cfile", dtype=np.complex64)
    assert len(tx) == 300000
    assert np.all(tx[:100000] == 0) and np.all(tx[-100000:] == 0)
    assert np.abs(tx).max() <= .005000001
    assert not (out/"launch.json").exists()
    with pytest.raises(ValueError, match="explicit"):
        DRIVER["capture"](out)
    assert not (out/"launch.json").exists()
    with pytest.raises(FileExistsError):
        DRIVER["prepare"](out, build_manifest=fake_build, serial="0x1d423d9108f273", frequency_hz=500e6)


def test_fake_transport_rx_precedes_tx_and_receipts_preserve_bytes(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    result = DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    report = result["worker_report"]
    assert report["library_version"].endswith("FAKE-OFFLINE-TEST")
    assert result["success"] and result["cleanup_confirmed"]
    assert report["rx_ready_timestamp"] < report["tx_start_timestamp"]
    assert report["tx_samples_sent"] == 300000
    assert report["native_tx_gain_db"] == 0
    assert report["rx_lpf_hz"] == 1500000 and report["tx_lpf_hz"] == 5000000
    assert (out/"rx.cfile").stat().st_size == report["captured_samples"]*8
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    assert all((out/name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize("mode,tx_attempted", [("no_rx", False), ("gap", False), ("clip", False), ("tx_fail", True), ("antenna_fail", True)])
def test_worker_failure_halts_and_cleans_up(tmp_path, fake_build, monkeypatch, mode, tx_attempted):
    out = prepared(tmp_path, fake_build)
    monkeypatch.setenv("FAKE_MODE", mode)
    with pytest.raises(RuntimeError, match="smoke failed"):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    result = json.loads((out/"receipt.json").read_text())
    assert not result["success"] and result["cleanup_confirmed"]
    assert result["worker_report"]["tx_attempted"] is tx_attempted
    assert result["worker_report"]["cleanup"]["disable_tx_return"] == 0
    assert (out/"rx.cfile").exists()


def test_changed_frozen_waveform_refuses_before_launch(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    (out/"tx.cfile").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    assert not (out/"launch.json").exists()


def test_signal_after_tx_activation_records_cleanup(tmp_path, fake_build):
    out = prepared(tmp_path, fake_build)
    plan = DRIVER["load"](out)
    marker = tmp_path/"fake-tx-started"
    env = os.environ.copy()
    env.update(LD_PRELOAD=plan["library"], FAKE_MODE="slow", FAKE_TX_MARKER=str(marker))
    process = subprocess.Popen(plan["command"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    until = time.monotonic()+5
    while not marker.exists() and process.poll() is None and time.monotonic() < until:
        time.sleep(.01)
    assert marker.exists()
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=5)
    report = json.loads((out/"worker-status.json").read_text())
    assert not report["success"] and report["tx_attempted"]
    assert report["cleanup"]["disable_tx_return"] == 0
    assert report["cleanup"]["antenna_off_readback"] == 0


def test_parent_exception_stops_launched_worker_before_receipt(tmp_path, fake_build, monkeypatch):
    out = prepared(tmp_path, fake_build)
    monkeypatch.setenv("FAKE_MODE", "slow")
    real_popen = subprocess.Popen
    children = []

    def injected_parent_failure(_seconds):
        raise RuntimeError("injected parent failure")

    real_sleep = time.sleep
    def tracked_with_real_wait(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        real_sleep(.03)
        return child

    monkeypatch.setattr(subprocess, "Popen", tracked_with_real_wait)
    monkeypatch.setitem(DRIVER["capture"].__globals__, "time",
                        SimpleNamespace(monotonic=time.monotonic, sleep=injected_parent_failure))
    with pytest.raises(RuntimeError, match="injected parent failure"):
        DRIVER["capture"](out, transmit=True, rf_confined_authorized=True)
    assert len(children) == 1 and children[0].poll() is not None
    receipt = json.loads((out/"receipt.json").read_text())
    assert "injected parent failure" in receipt["launch_error"]
    assert not receipt["success"] and not receipt["forced_kill"]
    assert receipt["cleanup_confirmed"]
