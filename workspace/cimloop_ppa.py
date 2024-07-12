from __future__ import annotations
from .scripts import get_spec_hd, run_mapper
import os
import joblib
from typing import List, Tuple, Dict, Any
import yaml
from torch import Tensor, nn
import torch
import math
import numpy as np

EXTRA_COMPONENT = """
- !Component # Column readout (ADC)
  name: {}
  <<<: [*component_defaults, *keep_outputs, *no_coalesce]
  subclass: dummy_storage
  attributes: {{width: ENCODED_OUTPUT_BITS, n_bits: width, <<<: *cim_component_attributes}}
"""

# NeuroSim has some extra digital components. For fair comparison, we'll add as
# many components it has. We're not using them for energy so we'll just realize
# all of them as intadders.
EXTRA_NEUROSIM_COMPONENTS = ["shift_add", "adder", "pooling", "activation"]
EXTRA_COMPONENTS_CONFIG = "\n".join(
    EXTRA_COMPONENT.format(name) for name in EXTRA_NEUROSIM_COMPONENTS
)

def normalize_tensor(tensor: Tensor) -> List[float]:
    flattened_tensor = tensor.flatten()
    if min_value := flattened_tensor.min() < 0:
        processed_tensor = flattened_tensor - min_value
    else:
        processed_tensor = flattened_tensor
    
    tensor_sum = processed_tensor.sum()
    normalized_tensor = processed_tensor / tensor_sum
    normalized_list = normalized_tensor.tolist()
    assert math.isclose(sum(normalized_list), 1.0), f"Sum of normalized list is not 1.0: {sum(normalized_list)}"
    return normalized_list

def process_data(data: List[float]) -> float:
    d = np.array(data)
    if np.min(d) < 0:
        ret = (d + 1) / 2
    else:
        ret = d
    return np.mean(ret)

def get_layer_data(model: nn.Module, input_tensor: Tensor):
    layer_data: List[Dict[str, Any]] = []

    def hook_fn(module: nn.Module, input: Tensor, output: Tensor):
        assert hasattr(module, 'name'), f"Module {module} does not have a name attribute"
        assert hasattr(module, 'weight'), f"Module {module} does not have a weight attribute"

        layer_info = {
            'name': module.name,
            'Inputs': input,
            'Weight': module.weight,
            'Outputs': output,
            'instance': {"C": module.in_features, "M": module.out_features}
        }
        layer_data.append(layer_info)

    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return layer_data


def write_layer(filename: str, instance: Dict[str, int], name: str, dnn_name: str, inputs: List[float], weights: List[float], outputs: List[float]):
    data = {
        'problem': {
            '<<': "*problem_base",
            'instance': instance,
            'name': name,
            'dnn_name': dnn_name,
            'notes': name,
            'histograms': {
                "Inputs": inputs,
                "Weights": weights,
                "Outputs": outputs,
            }
        }
    }

    yaml_content = f"{{{{include_text('../problem_base.yaml')}}}}\n" + yaml.dump(data, default_flow_style=False, sort_keys=False)

    with open(filename, 'w') as file:
        file.write(yaml_content)

def run_layer(
    dnn: str,
    layer: str,
    avg_input: float,
    avg_weight: float,
    shape: tuple,
    reram_size: int,
    max_mappings: int | None = None,
):
    spec = get_spec_hd(
        "basic_analog",
        system="ws_dummy_buffer_many_macro",
        dnn=dnn,  # Set the DNN and layer
        layer=layer,
        jinja_parse_data={
            "cell_override": "rram_neurosim_default.cell.yaml",
            "ignoreme_placeholder": EXTRA_COMPONENTS_CONFIG,
        },
    )

    # NeuroSim's default macro variable settings
    spec.variables.update(
        dict(
            INPUT_ENCODING_FUNC="offset_encode_if_signed_hist",
            WEIGHT_ENCODING_FUNC="offset_encode_if_signed_hist",
            VOLTAGE=0.85,
            TECHNOLOGY=22,  # nm
            BITS_PER_CELL=2,
            ADC_RESOLUTION=5,
            VOLTAGE_DAC_RESOLUTION=1,
            TEMPORAL_DAC_RESOLUTION=1,
            N_SHIFT_ADDS_PER_BANK=16,
            N_ADC_PER_BANK=16,
            BASE_LATENCY=1e-12,  # For near-zero leakage, make it really fast.
            READ_PULSE_WIDTH=1e-8,
            VOLTAGE_ENERGY_SCALE=1,
            VOLTAGE_LATENCY_SCALE=1,
            AVERAGE_INPUT_VALUE=float(avg_input),
            AVERAGE_WEIGHT_VALUE=float(avg_weight),
            BATCH_SIZE=1,
        )
    )
    spec.architecture.find("row").spatial.meshY = reram_size
    spec.architecture.find("column").spatial.meshX = reram_size
    spec.architecture.find("adc").attributes[
        "adc_estimator_plug_in"
    ] = '"Neurosim Plug-In"'

    # Set the shape of the layer. NeuroSim uses a different shape than Timeloop
    spec.variables["MAX_UTILIZATION"] = False
    ins = spec.problem.instance
    ins["P"] = (shape[0] - shape[3] + 1) // shape[7]
    ins["Q"] = (shape[1] - shape[4] + 1) // shape[7]
    ins["C"] = shape[2]
    ins["R"] = shape[3]
    ins["S"] = shape[3]
    ins["M"] = shape[5]
    ins["WStride"] = shape[7]
    ins["HStride"] = shape[7]

    # Lock in the mapping to only evaluate one mapping by defualt
    dt = spec.architecture.find("dummy_top").constraints.temporal
    dt.factors_only = dt.factors
    dt.factors.clear()
    dt.factors_only.add_eq_factor("X", 8)
    dt.factors_only.add_eq_factor("P", -1)
    dt.factors_only.add_eq_factor("Q", -1)
    spec.mapping.max_permutations_per_if_visit = 1

    # If there's a max_mappings, expand the search space
    if max_mappings is not None:
        # Set to evaluate max_mappings mappings
        spec.mapper.search_size = max_mappings
        spec.mapper.max_permutations_per_if_visit = max_mappings
        spec.mapper.victory_condition = max_mappings

        # Expand the problem and relax constraints to expand the search space
        for d in "RSMCPQXYZG":
            spec.problem.instance[d] = 8
        for t in ["row", "column", "macro", "dummy_top", "1bit_x_1bit_mac"]:
            t = spec.architecture.find(t)
            t.constraints.spatial.permutation.clear()
            t.constraints.temporal.permutation.clear()

    # Some specific mappings needed to match NeuroSim. Timeloop will find a
    # lower-energy mapping if we don't set these.
    if ins["C"] == 64:
        rs = spec.architecture.find("row").constraints.spatial
        rs.factors.clear()
        rs.factors_only = rs.factors
        rs.factors_only.add_eq_factor("C", 64)
        rs.factors_only.add_eq_factor("R", 1)
        rs.factors_only.add_eq_factor("S", 1)
        rs.maximize_dims = None

    # Run the mapper
    return run_mapper(spec)

def write_model(model_name: str, layer_data: List[Dict[str, Any]]) -> None:
    for layer in layer_data:
        name = layer['name']
        inputs = normalize_tensor(layer['Inputs'])
        weights = normalize_tensor(layer['Weight'])
        outputs = normalize_tensor(layer['Outputs'])
        instance = layer['instance']
        write_layer(f"models/workloads/{model_name}/{name}.yaml", instance, name, model_name, inputs, weights, outputs)

def get_averages(layer_data: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[Tuple[int, ...]]]:
    input_averages: List[float] = [process_data([i['Inputs'] for i in layer_data])]
    weight_averages: List[float] = [process_data([i['Weight'] for i in layer_data])]

    def get_shape(layer: Dict[str, int]) -> Tuple[int, ...]:
        return (1, 1, layer["C"], 1, 1, layer["M"], 0, 1)

    SHAPES: List[Tuple[int, ...]] = [get_shape(i['instance']) for i in layer_data]

    return input_averages, weight_averages, SHAPES

def cimloop_ppa(model_name: str, model: nn.Module, x_test: Tensor, ram_size: int, frequency: int, temperature: int, cell_bit: int):
    layer_data = get_layer_data(model, x_test)
    write_model(model_name, layer_data)
    input_averages, weight_averages, SHAPES = get_averages(layer_data)
    
    layers = [f for f in os.listdir(f"models/workloads/{model_name}") if f != "index.yaml" and f.endswith(".yaml")]
    layers = sorted(l.split(".")[0] for l in layers)

    # # CiMLoop One Mapping
    results = joblib.Parallel(n_jobs=32)(
        joblib.delayed(run_layer)(model_name, layer, avg_input, avg_weight, shape, ram_size)
        for layer, avg_input, avg_weight, shape in zip(
            layers, input_averages, weight_averages, SHAPES
        )
    )

    # CiMLoop 10k mappings
    # results = joblib.Parallel(n_jobs=32)(
    #     joblib.delayed(run_layer)(DNN, layer, avg_input, avg_weight, shape, 10000)
    #     for layer, avg_input, avg_weight, shape in zip(
    #         layers, input_averages, weight_averages, SHAPES
    #     )
    # )

    for result in results: 
        pass
    return list(results)