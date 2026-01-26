"""
Test with Database - Run perspective service with real DB connection and JSON input.

Usage:
    python test_with_database.py <input_json_file> [--verbose]
"""

import sys
import io
import json
import argparse
from time import perf_counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from config import load_config
from perspective_service.core.engine import PerspectiveEngine


def main():
    parser = argparse.ArgumentParser(description='Test perspective service with JSON input')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Include removal summary')
    parser.add_argument('--flatten', '-f', action='store_true', help='Flatten output')
    args = parser.parse_args()

    # Load config
    config = load_config()
    connection_string = config.get_odbc_connection_string()

    # Load input
    with open(args.input_file, 'r') as f:
        request = json.load(f)

    perspective_configs = request.get('perspective_configurations', {})
    position_weights = request.get('position_weight_labels', ['weight'])
    lookthrough_weights = request.get('lookthrough_weight_labels', ['weight'])
    system_version_timestamp = request.get('system_version_timestamp')

    # Initialize engine
    print("Initializing engine...")
    engine = PerspectiveEngine(
        connection_string=connection_string,
        system_version_timestamp=system_version_timestamp
    )
    print(f"Loaded {len(engine.config.perspectives)} perspectives, {len(engine.config.modifiers)} modifiers")

    # Process
    print("\nProcessing...")
    start = perf_counter()

    result = engine.process(
        input_json=request,
        perspective_configs=perspective_configs,
        position_weights=position_weights,
        lookthrough_weights=lookthrough_weights,
        verbose=args.verbose,
        flatten_response=args.flatten
    )

    elapsed = perf_counter() - start
    print(f"Processing complete in {elapsed*1000:.2f}ms")

    # Output summary
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)

    configs = result.get('perspective_configurations', {})
    for config_name, perspectives in configs.items():
        print(f"\nConfig: {config_name}")
        for pid, containers_data in perspectives.items():
            print(f"  Perspective {pid}:")
            for container, data in containers_data.items():
                positions = data.get('positions', {})
                removed = data.get('removed_positions_weight_summary', {})
                lt_keys = [k for k in data.keys() if 'lookthrough' in k and k != 'removed_positions_weight_summary']
                lt_count = sum(len(data.get(k, {})) for k in lt_keys)

                print(f"    {container}:")
                print(f"      Positions kept: {len(positions)}")
                print(f"      Lookthroughs kept: {lt_count} (keys: {lt_keys})")
                if removed:
                    for rtype, ritems in removed.items():
                        print(f"      Removed {rtype}: {len(ritems)} items")

    # Full output
    print("\n" + "=" * 60)
    print("FULL OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
