import argparse
import os
import importlib.util
import sys

import circuit_level_css
import circuit_level_linear
import circuit_level_xy

spec = importlib.util.spec_from_file_location("circuit_level_ti", "circuit_level_ti_(0.25,0.5).py")
circuit_level_ti = importlib.util.module_from_spec(spec)
sys.modules["circuit_level_ti"] = circuit_level_ti
spec.loader.exec_module(circuit_level_ti)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=6)
    parser.add_argument("--m", type=int, default=6)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--bias", type=float, default=10000.0)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="distance_check")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    modules = {
        "css": circuit_level_css,
        "linear": circuit_level_linear,
        "ti": circuit_level_ti,
        "xy": circuit_level_xy
    }

    for name, mod in modules.items():
        if name == "ti":
            circuit = mod.generate_circuit(
                "tile_code:memory_x",
                rounds=args.rounds,
                x_distance=args.l,
                z_distance=args.m,
                before_round_data_depolarization=args.p,
                before_measure_flip_probability=args.p,
                after_reset_flip_probability=args.p,
                after_clifford_depolarization=args.p,
                after_single_clifford_probability=args.p,
                bias=args.bias,
            )
        else:
            circuit = mod.generate_circuit(
                rounds=args.rounds,
                x_distance=args.l,
                z_distance=args.m,
                before_round_data_depolarization=args.p,
                before_measure_flip_probability=args.p,
                after_reset_flip_probability=args.p,
                after_clifford_depolarization=args.p,
                after_single_clifford_probability=args.p,
                bias=args.bias,
            )
        out_path = os.path.join(args.outdir, f"circuit_{name}_l{args.l}_m{args.m}.stim")
        circuit.to_file(out_path)
        print(f"Generated {out_path}")

if __name__ == "__main__":
    main()
