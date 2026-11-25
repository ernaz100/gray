# Clusters

One file per cluster says where things are there: the repository, the data folder, the Python
interpreter, the options a GPU job needs, the partitions to try, and settings for its GPUs.
`jobs/submit.sh` (run on the workstation) reads them, so a training is submitted with the same
command anywhere:

    jobs/submit.sh ask     4 motion ee-motion-v1 --ee4d @data/ee4d     # when would each cluster start it?
    jobs/submit.sh fastest 4 motion ee-motion-v1 --ee4d @data/ee4d     # submit where it starts soonest
    jobs/submit.sh viper   4 motion ee-motion-v1 --ee4d @data/ee4d

`@data` is replaced by the cluster's data folder. Trainings run as chains of 1-hour jobs that
resume one another; the run folder is `<data>/runs/<run name>` on the cluster that ran it.

## What is where

| Data | Size | Viper | Raven | MPI |
|---|---:|---|---|---|
| Nymeria: features, image grids, moments, maps, body | 700 GB | yes | no | no |
| `ee4d/` (UniEgoMotion's data in our form) | 13 GB | yes | no | no |
| `hml3d/` (HumanML3D in our form) | 2 GB | yes | no | no |

Trainings on Nymeria can only run on Viper. Those on `ee4d` and `hml3d` can run wherever those
folders have been copied to.

## Adding a cluster, or data to one

1. Clone the repository to the place its file names as `GRAY_CODE` (`git clone https://github.com/ernaz100/gray`).
2. Check that `GRAY_PYTHON` exists and has torch for that cluster's GPUs.
3. Copy the data the trainings need into `GRAY_DATA`. The clusters cannot reach one another; copy from
   the workstation, where `gray/local/data` holds `ee4d` and `hml3d`.
4. `jobs/submit.sh ask ...` then lists the cluster with a start time.

A run that started on one cluster continues there: its saved state is in that cluster's `runs` folder.
