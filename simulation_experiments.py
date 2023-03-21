import os
import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import pickle
import cvxpy

from geometry_utils import triangulation

default_solver_params = {
    "SCS": {"eps_abs": 1e-12, "eps_rel": 1e-12},
    "MOSEK": {
        "mosek_params": {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-14}
    },
}


def solve_problem(problem, observations, method, solver='MOSEK'):
    sdr = method(problem.poses, observations, problem.K, scale=1 / 1054.)

    solver_params = default_solver_params[solver]

    try:
        t_start = time.time()
        sdr.solve(solver=solver, warm_start=False, **solver_params)
        t = t_start - time.time()
        results = sdr.get_solution(eps=1e-4)
        cost = triangulation.robust_cost(
            point=results["estimated_point"],
            poses=problem.poses,
            observations=observations,
            intrinsics=problem.K,
            c=sdr.c / sdr.scale ** 2
        )
        sdr_objective = sdr.objective() / sdr.scale ** 2
        error = np.linalg.norm(results["estimated_point"] - problem.point)
    except cvxpy.error.SolverError:
        results = {}
        t = np.inf
        cost, error, sdr_objective = None, None, None

    return {
        'problem': problem,
        'n_poses': problem.n_poses,
        'observations': observations,
        **results,
        'time': t,
        'results': results,
        'cost': cost,
        'error': error,
        'sdr_objective': sdr_objective,
        'solver': solver,
        'X': sdr.X.value,
        **solver_params,
    }


def itemtqdm(items, desc, *args, **kwargs):
    progress_bar = tqdm(items, *args, **kwargs)
    for item in progress_bar:
        progress_bar.set_description(f'{desc} ({item})')
        yield item


def save_results(results, output_dir, file_index):
    results = pd.DataFrame(results)
    with open(output_dir / f"results{file_index}.pickle", "wb") as pickle_file:
        pickle.dump(results, pickle_file)


def load_pickle(file_name):
    with open(file_name, 'rb') as pfile:
        return pickle.load(pfile)


def load_results(base_dir):
    n_files = len([f for f in os.listdir(base_dir) if 'results' in f])
    results = pd.concat([load_pickle(Path(base_dir) / f'results{i}.pickle') for i in range(n_files)])
    return results


METHODS = {
    # "epipolar": triangulation.TriangulationSDR,
    "robust_epipolar": triangulation.RobustTriangulationSDR,
    # "fractional": triangulation.TriangulationFractionalSDR,
    # "robust_fractional": triangulation.RobustTriangulationFractionalSDR,
    # "fractional_partial": triangulation.TriangulationFractionalSDR,
    # "robust_fractional_partial": triangulation.RobustTriangulationFractionalSDR,
}


def run(output_dir: Path = Path('results/simulated'), timestamped_output_dir: bool = True):
    if timestamped_output_dir:
        output_dir = output_dir / datetime.now().strftime(r'%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=False)
    n_trials = 30

    height, width = 1162, 2108
    K = np.array([
        [1012.0027, 0, 1054],
        [0., 1012.0027, 581],
        [0., 0., 1.],
    ])

    results = []
    file_index = 0
    for n_poses in itemtqdm([3, 5, 7], desc='n_poses'):
        # for n_poses in itemtqdm([25, 30], desc='n_poses'):
        outliers = range(n_poses - 1)
        # outliers = {
        #     25: [0, 10, 20],
        #     30: [0, 10, 20, 25],
        # }[n_poses]
        for n_outliers in itemtqdm(outliers, leave=False, desc='n_outliers'):
            for observation_noise in itemtqdm(np.linspace(0., 100, 10), leave=False, desc='noise'):
                for trial_idx in itemtqdm(range(n_trials), leave=False, desc='trial_idx'):
                    problem = triangulation.get_aholt_problem(n_poses, mode="sphere", K=K, radius=2)
                    observations = problem.get_observations(sigma=observation_noise)
                    corrupted_observations, inlier_mask = triangulation.add_outliers(
                        observations, n_outliers, height, width, check_reprojection=False
                    )
                    for method_name, method in tqdm(METHODS.items(), desc='method', leave=False):
                        for c in tqdm([200], leave=False, desc='c'):
                            for with_inequalities in [True, False]:
                                for solver in ["MOSEK"]:
                                    results.append({
                                        'observation_noise': observation_noise,
                                        'method': method_name,
                                        'n_poses': n_poses,
                                        'inlier_mask': inlier_mask,
                                        'n_outliers': n_outliers,
                                        'trial_index': trial_idx,
                                        'c': c,
                                        'with_inequalities': with_inequalities,
                                        **solve_problem(problem, corrupted_observations,
                                                        method=lambda *args, **kwargs: method(
                                                            *args, **kwargs, c=np.ones(n_poses) * c ** 2,
                                                            with_inequalities=with_inequalities,
                                                            # full_constraints='partial' not in method_name,
                                                        ), solver=solver),
                                    })
                                    if len(results) >= 100:
                                        save_results(results, output_dir, file_index)
                                        results = []
                                        file_index += 1
    save_results(results, output_dir, file_index)


if __name__ == '__main__':
    run()
