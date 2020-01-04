import numpy as np

import time
import random
import pytest

from simulated_annealing.model import model


def test_parameter_initialization():
    parameters = model.ProblemSpec()

    # Verify shape and ensure padding of the occupancy array.
    assert len(parameters.fixed_preference_costs) == len(
        parameters.variable_preference_costs) == parameters.preferences

    assert parameters.preference_costs.shape[1] == \
           parameters.period + parameters.padding

    assert len(parameters.preference_delta) <= (parameters.period * \
                                                parameters.period - 1) * len(
        parameters.family_size_pool)

    # Test the accuracy of the account penalty function. Ensure the incremental
    # updates to the schedule cost is equivalent to calculating the
    # accounting penalty on the entire schedule.

    annealer = model.Annealer(parameters)


    precision = 0
    trials = 10
    perturbations = 3
    delta_time = 0
    macro_time = 0
    for trial in range(0,trials):
        schedule1 = annealer.rand_init()
        for sample in range(0, perturbations):

            while True:
                days = random.sample(range(1, 101), 2)
                family_id = random.choice(range(0, parameters.families))
                if schedule1.schedule[family_id][days[0]] == 0 and \
                        schedule1.schedule[family_id][days[1]] == 1:
                    family_size = parameters.family_sizes[family_id]
                    n1 = schedule1.occupancy[0][days[0]] + family_size
                    n2 = schedule1.occupancy[0][days[1]] - family_size
                    if 125 <= n1 <= 300 and 125 <= n2 <= 300:
                        update = model.ScheduleUpdate(parameters,
                                                      [[family_id] + days])
                        schedule1.update(update)
                        break

        cost1 = schedule1.schedule_cost

        schedule2 = model.Schedule(parameters, schedule1.schedule)
        #schedule2.update()
        cost2 = schedule2.schedule_cost
        print("cost delta", cost1, cost2)
        error = abs((cost2 - cost1))/cost2
        print(error)
        if error <= 1:
            precision += 1

    assert precision == trials
#
# def test_schedule_initialization():
#     file_path = "/data/family_data.csv"
#     parameters = model.ScheduleParams(file_path)
#     schedule = model.Schedule(parameters)
#     schedule2 = model.Schedule(parameters)
#
#     # Check that the schedule satisfies constraints.
#
#
#     assert ((schedule.occupancy[0] >= 125).all() and (schedule.occupancy[0] <=
#                                                      300).all()) == True
#
#     # Test update_occupancy method agrees if argument is supplied
#
#
#
#     samples = 10
#     agrees = False
#     for _ in range(0, samples):
#
#         occupancy_schedule = np.zeros((2, 100))
#         occupancy_schedule[0] = np.random.randint(125, 300, 100)
#
#         for day in range(0, parameters.days - 1):
#             occupancy_schedule[1][day] = \
#                 abs(occupancy_schedule[0][day + 1] - occupancy_schedule[0][
#                     day])
#         mod_days = random.sample(range(0, parameters.days), random.choice(
#             range(0,parameters.days)))
#
#         for day in mod_days:
#             upper = 300 - int(occupancy_schedule[0][day])
#             lower = 125 - int(occupancy_schedule[0][day])
#             delta = random.choice(range(lower, upper + 1))
#             occupancy_schedule[0][day] += delta
#
#         schedule.occupancy = np.copy(occupancy_schedule)
#         schedule2.occupancy = np.copy(occupancy_schedule)
#
#         schedule.update_occupancy()
#         schedule2.update_occupancy(days=mod_days)
#
#         if not (schedule.occupancy[1] == schedule2.occupancy[1]).all():
#             print("heeere")
#             break
#         agrees = True
#
#     assert agrees
#
