import numpy as np
import ray
from ray.tune import Trainable, run
from ray.tune.schedulers import PopulationBasedTraining

class SATrialScheduler(Trainable)

    def _setup(self):
        pass

    def _train(self):
        pass


    def _save(self):
        pass

    def _restore(self):
        pass

if __name__ == '__main__':

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="schedule_cost",
        mode="max",
    )

    run(
        SATrialScheduler,
        name='sa_parameter_search',
        scheduler=pbt,
        reuse_actors=True,


    )
