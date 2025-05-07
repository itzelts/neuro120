# -*- coding: utf-8 -*-
"""
Main script to run all enhanced analyses for the motor task fMRI data
"""
import pathlib
import sys
import time

def main():
    # Create results directory
    results_dir = pathlib.Path('enhanced_results')
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Running Enhanced Analysis Pipeline")
    print("=" * 80)
    
    # Record start time
    start_time = time.time()
    
    # Run Analysis 1: Graded Functional Similarity
    print("\nPart 1: Graded Functional Similarity Analysis")
    print("-" * 80)
    try:
        from enhanced_analysis import run_graded_analysis
        run_graded_analysis()
    except Exception as e:
        print(f"Error in Graded Functional Analysis: {e}")
    
    # Run Analysis 2: Alternative Functional Groupings
    print("\nPart 2: Alternative Functional Groupings Analysis")
    print("-" * 80)
    try:
        sys.path.append('.')
        from enhanced_analysis_part2 import run_alternative_groupings_analysis
        run_alternative_groupings_analysis()
    except Exception as e:
        print(f"Error in Alternative Functional Groupings Analysis: {e}")
    
    # Run Analysis 3: Individual Variability & Data-Driven Refinement
    print("\nPart 3: Individual Variability & Data-Driven Refinement")
    print("-" * 80)
    try:
        from enhanced_analysis_part3 import run_all_analyses
        run_all_analyses()
    except Exception as e:
        print(f"Error in Individual Variability Analysis: {e}")
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print(f"Enhanced Analysis Pipeline completed in {duration:.2f} seconds")
    print("=" * 80)
    print(f"\nResults saved to: {results_dir.absolute()}")

if __name__ == "__main__":
    main()
