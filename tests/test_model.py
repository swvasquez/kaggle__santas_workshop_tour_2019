import numpy as np

import time
import random
import pytest

from simulated_annealing.model import model


def test_parameter_initialization():
    file_path = "/data/family_data.csv"
    parameters = model.ScheduleParams(file_path)

    # Verify shape and ensure padding of the occupancy array.
    assert parameters.preference_costs.shape == (5000, 100)

    assert np.count_nonzero(parameters.preference_costs) == 5000 * 99

    # Test the accuracy of the account penalty function. Ensure the incremental
    # updates to the schedule cost is equivalent to calculating the
    # accounting penalty on the entire schedule.

    occupancy_schedule = np.zeros((2, 100))
    occupancy_schedule[0] = np.random.randint(125, 300, 100)

    for day in range(0, parameters.days - 1):
        occupancy_schedule[1][day] = \
            abs(occupancy_schedule[0][day + 1] - occupancy_schedule[0][day])

    precision = 0
    samples = 10
    delta_time = 0
    macro_time = 0

    for sample in range(0, samples):
        original_cost = parameters.accounting_penalty(occupancy_schedule[0],
                                                      occupancy_schedule[1])
        while True:
            days = random.sample(range(0, 100), 2)
            family_size = random.choice(parameters.family_sizes_pool)
            n1 = occupancy_schedule[0][days[0]] + family_size
            n2 = occupancy_schedule[0][days[1]] - family_size
            if 125 <= n1 <= 300 and 125 <= n2 <= 300:
                break

        start_t = time.perf_counter()
        delta_cost = parameters.ap_delta(days, family_size,
                                         occupancy_schedule)
        end_t = time.perf_counter()
        delta_time += end_t - start_t

        start_t = time.perf_counter()

        occupancy_schedule[0][days[0]] = n1
        occupancy_schedule[0][days[1]] = n2

        for day in range(0, 99):
            occupancy_schedule[1][day] = \
                abs(occupancy_schedule[0][day + 1] -
                    occupancy_schedule[0][day])

        macro_cost = parameters.accounting_penalty(
            occupancy_schedule[0],
            occupancy_schedule[1]) - original_cost


        end_t = time.perf_counter()

        macro_time += end_t - start_t
        error = abs((delta_cost - macro_cost))

        if error <= 1:
            precision += 1
        else:
            print(occupancy_schedule, "\n")

    print("Times", delta_time, macro_time)
    assert precision == samples

def test_schedule_initialization():
    file_path = "/data/family_data.csv"
    parameters = model.ScheduleParams(file_path)
    schedule = model.Schedule(parameters)
    schedule2 = model.Schedule(parameters)

    # Check that the schedule satisfies constraints.


    assert ((schedule.occupancy[0] >= 125).all() and (schedule.occupancy[0] <=
                                                     300).all()) == True

    # Test update_occupancy method agrees if argument is supplied



    samples = 10
    agrees = False
    for _ in range(0, samples):

        occupancy_schedule = np.zeros((2, 100))
        occupancy_schedule[0] = np.random.randint(125, 300, 100)

        for day in range(0, parameters.days - 1):
            occupancy_schedule[1][day] = \
                abs(occupancy_schedule[0][day + 1] - occupancy_schedule[0][
                    day])
        mod_days = random.sample(range(0, parameters.days), random.choice(
            range(0,parameters.days)))

        for day in mod_days:
            upper = 300 - int(occupancy_schedule[0][day])
            lower = 125 - int(occupancy_schedule[0][day])
            delta = random.choice(range(lower, upper + 1))
            occupancy_schedule[0][day] += delta

        schedule.occupancy = np.copy(occupancy_schedule)
        schedule2.occupancy = np.copy(occupancy_schedule)

        schedule.update_occupancy()
        schedule2.update_occupancy(days=mod_days)

        if not (schedule.occupancy[1] == schedule2.occupancy[1]).all():
            print("heeere")
            break
        agrees = True

    assert agrees



