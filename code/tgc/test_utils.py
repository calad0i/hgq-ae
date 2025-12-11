from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import keras
import model as _  # noqa: F401 # type: ignore
import numpy as np
from da4ml.codegen import RTLModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace
from da4ml.typing import solver_options_t
from hgq.utils import trace_minmax


def trace_and_save(model: keras.Model, path: str | Path, *datasets: Sequence[np.ndarray]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ds, _dss = datasets[0], datasets[1:]
    trace_minmax(model, _ds, batch_size=25600, reset=True)
    for ds in _dss:
        trace_minmax(model, ds, batch_size=25600, reset=False)
    model.save(path)


def convert_and_test(
    model: keras.Model,
    name: str,
    path: str | Path,
    ds_test: tuple[Any, Any] | None,
    metric: Callable[[np.ndarray, np.ndarray], float | Sequence[float]] | None,
    sw_test: bool,
    hw_test: bool,
    hw_config: HWConfig = HWConfig(1, 8, -1),
    latency_cutoff=8,
    solver_options: solver_options_t | None = None,
    clock_period: float = 5.0,
    clock_uncertainty: float = 0.0,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inp, out = trace_model(model, hwconf=hw_config, solver_options=solver_options)
    comb = comb_trace(inp, out)
    rtl = RTLModel(
        comb, name, path, latency_cutoff=latency_cutoff, clock_period=clock_period, clock_uncertainty=clock_uncertainty
    )

    ebops = 0
    for layer in model.layers:
        if getattr(layer, 'enable_ebops', False):
            ebops += int(layer.ebops)  # type: ignore
    misc: dict = {'ebops': ebops}

    rtl.write(misc)

    if not ds_test:
        return
    assert metric is not None, 'Metric must be provided if ds_test is given'
    y_true = ds_test[1]

    if sw_test:
        y_pred = model.predict(ds_test[0], batch_size=25600, verbose=0)  # type: ignore
        y_comb = comb.predict(ds_test[0], n_threads=4)  # type: ignore
        res = metric(y_true, y_pred)
        res_comb = metric(y_true, y_comb)

        misc['keras_metric'] = res
        misc['comb_metric'] = res_comb

        rtl.write(misc)

    if hw_test:
        for _ in range(8):
            try:
                rtl._compile(openmp=False, nproc=4)
                break
            except RuntimeError:
                pass

        y_pred_hw = rtl.predict(ds_test[0])
        if sw_test:
            assert np.all(y_pred_hw == y_comb), 'HW/SW output mismatch!'  # type: ignore
