import tyro
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime

from triangulation_relaxations import triangulation
from triangulation_relaxations.se3 import Se3
from simulation_experiments import solve_problem, itemtqdm, save_results

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat

METHODS = {
    # "epipolar": triangulation.TriangulationSDR,
    "robust_epipolar": triangulation.RobustTriangulationSDR,
    # "fractional": triangulation.TriangulationFractionalSDR,
    "robust_fractional": triangulation.RobustTriangulationFractionalSDR,
    # "fractional_partial": triangulation.TriangulationFractionalSDR,
    # "robust_fractional_partial": triangulation.RobustTriangulationFractionalSDR,
}

ydown2zup = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])


def get_intrinsics(camera):
    pars = camera.params
    return np.array([
        [pars[0], 0, pars[2]],
        [0, pars[1], pars[3]],
        [0, 0, 1]
    ])


def run(dataset_dir: Path, output_dir: Path = Path('results/image_matching'), timestamped_output_dir: bool = True):
    cameras, images, points = read_model(path=dataset_dir / 'dense/sparse', ext='.bin')

    view_inds = sorted(images.keys())
    r = np.array([qvec2rotmat(images[i].qvec) for i in view_inds])
    t = np.array([images[i].tvec for i in view_inds])
    poses = Se3(ydown2zup[:3, :3]) * Se3(r, t).inverse()
    intrinsics = np.array([get_intrinsics(cameras[images[i].camera_id]) for i in view_inds])
    shapes = np.array([[
        cameras[images[i].camera_id].height,
        cameras[images[i].camera_id].width
    ] for i in view_inds])

    print(f'Cameras: {len(cameras)}')
    print(f'Images: {len(images)}')
    print(f'3D points: {len(points)}')

    if timestamped_output_dir:
        output_dir =  output_dir / datetime.now().strftime(r'%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=False)

    file_index = 0
    image_ids = sorted(images.keys())
    results = []
    for n_views in itemtqdm([3, 5, 7], desc="n_views"):
        for method_name, method in tqdm(METHODS.items(), desc='method', leave=False):
            n_trials = 60 if 'fractional' in method_name else 375
            # n_trials = 120
            for trial_idx in itemtqdm(range(n_trials), desc='trial_idx', leave=False):
                point_id = np.random.choice(
                    [point_id for point_id, point in points.items() if len(point.image_ids) >= n_views])

                observation_ids = np.random.choice(range(len(points[point_id].image_ids)), n_views, replace=False)
                iamge_with_point_ids = points[point_id].image_ids[observation_ids]
                point2D_ids = points[point_id].point2D_idxs[observation_ids]
                pose_ids = [image_ids.index(image_id) for image_id in iamge_with_point_ids]

                problem = triangulation.TriangulationProblem(
                    poses[pose_ids],
                    ydown2zup[:3, :3] @ points[point_id].xyz,
                    intrinsics[pose_ids]
                )

                for n_outliers in tqdm(range(len(pose_ids) - 1), leave=False):
                    gt_point_observations = np.array([images[image_id].xys[point2D_id] for image_id, point2D_id in
                                                      zip(iamge_with_point_ids, point2D_ids)])
                    point_observations = gt_point_observations.copy()
                    outlier_inds = np.random.choice(range(n_views), n_outliers, replace=False)
                    if n_outliers > 0:
                        point_observations[outlier_inds] = np.array(
                            [images[image_id].xys[np.random.randint(len(images[image_id].xys))] for image_id in
                             iamge_with_point_ids[outlier_inds]])
                    for c in tqdm([10], leave=False, desc='c'):
                        for solver in ["MOSEK"]:
                            results.append({
                                'dataset': str(dataset_dir),
                                'pose_ids': pose_ids,
                                'method': method_name,
                                'n_views': n_views,
                                'inlier_mask': np.array([i not in outlier_inds for i in range(n_views)]),
                                'n_outliers': n_outliers,
                                'trial_index': trial_idx,
                                'c': c,
                                'shapes': shapes[pose_ids],
                                **solve_problem(problem, point_observations,
                                                method=lambda *args, **kwargs: method(
                                                    *args, **kwargs, c=np.ones(n_views) * c ** 2,
                                                ), solver=solver),
                            })
                            if len(results) >= 10:
                                save_results(results, output_dir, file_index)
                                results = []
                                file_index += 1
    save_results(results, output_dir, file_index)


if __name__ == '__main__':
    tyro.cli(run)
