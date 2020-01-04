import collections
import csv
import math
import pickle
import pathlib
import random
import json
import yaml

import numpy as np

from simulated_annealing import project_paths


class ProblemSpec:

    def __init__(self):

        paths = project_paths()
        root = paths['root']
        config_path = paths['config']

        with config_path.open(mode='r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        init = config['init']
        data = config['data']

        self.families = int(init['families'])
        self.period = int(init['period'])
        self.min_occupancy = int(init['min_occupancy'])
        self.max_occupancy = int(init['max_occupancy'])
        self.preferences = int(init['preferences'])
        self.fixed_preference_costs = np.array([int(cost) for cost in init[
            'fixed_preference_costs']])
        self.variable_preference_costs = np.array([int(cost) for cost in init[
            'variable_preference_costs']])
        self.pickle_segments = int(data['pc_pickle_segments'])

        self.family_ids = np.arange(self.families)
        self.family_sizes = np.arange(self.families)
        self.padding = 2
        self.preference_costs = np.zeros((self.families, self.period +
                                          self.padding))

        self.family_size_pool = []
        self.preference_delta = {}

        self._parse_family_preferences_csv(root / data[
            'preference_csv'])
        self.pc_pickle_dir = root / data['pc_pickle_dir']
        file_count = sum(1 for file in self.pc_pickle_dir.iterdir() if
                         file.stem.startswith('pc_delta_pickle'))
        if file_count in [int(data['pc_pickle_segments']),
                          int(data['pc_pickle_segments']) + 1]:

            self._load_pickled_pc_dict()
        else:
            self._gen_pref_delta_dict()

    def _parse_family_preferences_csv(self, path):

        family_sizes = np.zeros(self.families)
        family_size_pool = []
        preference_costs = np.zeros((self.families, self.period +
                                     self.padding))

        with path.open(mode='r') as reader:
            csv_reader = csv.reader(reader)
            next(csv_reader)
            for row in csv_reader:
                family_id = int(row[0])
                family_size = int(row[11])
                family_sizes[family_id] = family_size
                preference_costs[family_id] = \
                    self.fixed_preference_costs[-1] + family_size * \
                    self.variable_preference_costs[-1]
                for col in range(1, 11):
                    day = int(row[col])
                    preference_costs[family_id, day] = \
                        self.fixed_preference_costs[col - 1] + \
                        family_size * self.variable_preference_costs[col - 1]

        family_size_pool += list(set(family_sizes))

        self.family_size_pool = family_size_pool
        self.family_sizes = family_sizes
        self.preference_costs = preference_costs

    def _gen_pref_delta_dict(self):

        preference_delta = {}
        preference_delta_seg = {}
        pickled = 0
        family_sizes = set()
        for day0 in range(1, self.period + 1):
            for day1 in range(1, self.period + 1):
                print(day0, day1)
                if day0 != day1:
                    for family_id in self.family_ids:
                        # TODO: Determine order of subtraction.
                        delta = self.preference_costs[family_id][day0] - \
                                self.preference_costs[family_id][day1]
                        key = (day0, day1, int(self.family_sizes[family_id]))
                        family_sizes.add(int(self.family_sizes[family_id]))
                        if key in preference_delta:
                            preference_delta_seg[key].append(
                                [int(family_id), delta])
                        else:
                            preference_delta_seg[key] = [[int(family_id),
                                                          delta]]

                    for key in ((day0, day1, size) for size in family_sizes):
                        preference_delta_seg[key].sort(key=lambda x: x[1])
                    family_sizes = set()

                if day0 % (self.period // self.pickle_segments) == 0 and \
                        day1 == self.period or day0 == day1 == self.period:
                    p_path = self.pc_pickle_dir / f"pc_delta_pickle_{pickled}"
                    with p_path.open(mode='wb') as pc_pickle:
                        pickle.dump(preference_delta_seg, pc_pickle)
                    preference_delta = {**preference_delta,
                                        **preference_delta_seg}
                    preference_delta_seg = {}
                    pickled += 1

        self.preference_delta = preference_delta

    def _load_pickled_pc_dict(self):
        preference_delta = {}
        for file in self.pc_pickle_dir.iterdir():
            if file.stem.startswith('pc_delta_pickle'):
                with file.open(mode='rb') as pickle_binary:
                    preference_delta = {**pickle.load(pickle_binary),
                                        **preference_delta}
        self.preference_delta = preference_delta

    @staticmethod
    def accounting_penalty(daily_occupancy, daily_shift):

        accounting_penalty = np.sum(
            .0025 * (daily_occupancy - 125) *
            daily_occupancy ** (1 / 2 + daily_shift / 50))

        return accounting_penalty

    def preference_cost(self, schedule):
        preference_cost = np.sum(self.preference_costs * schedule)
        return preference_cost

    def cost(self, schedule):
        ap = self.accounting_penalty(schedule.occupancy[0],
                                     schedule.occupancy[1])
        pc = self.preference_cost(schedule.schedule)

        return ap + pc

    def ap_delta(self, schedule, occupancy_update):
        schedule_occupancy = np.copy(schedule.occupancy)

        print("types", schedule.occupancy[0].dtype)
        schedule.occupancy[0] += occupancy_update.table

        indices = set(occupancy_update.days) | \
                  {day - 1 for day in occupancy_update.days}

        for idx in indices:
            schedule_occupancy[1][idx] = abs(schedule_occupancy[0][idx + 1]
                                             - schedule_occupancy[0][idx])

        new_cost = self.accounting_penalty(
            schedule_occupancy[[0], list(indices)],
            schedule_occupancy[[1], list(indices)])

        old_cost = self.accounting_penalty(schedule.occupancy[[0],
                                                              list(indices)],
                                           schedule.occupancy[[1],
                                                              list(indices)])

        ap_delta = new_cost - old_cost

        return ap_delta

    def pc_delta(self, update):
        pref_delta = \
            np.sum(self.preference_costs[update.days][update.families] *
                   update.table)
        return pref_delta

    def delta_cost(self, update):

        delta_cost = self.pc_delta(update.occupany_update) + self.ap_delta(
            update)

        return delta_cost


class OccupancyUpdate:
    def __init__(self, parameters, updates=None):
        if updates is None:
            updates = []
        self.days = set()
        self.count = len(updates)
        self.table = np.zeros(parameters.period + parameters.padding)

        for update in updates:
            print(update)
            self.days |= {update[1], update[2]}
            self.table[update[1]] += update[0]
            self.table[update[2]] += -update[0]


class ScheduleUpdate:
    def __init__(self, parameters, updates=None):
        if updates is None:
            updates = []
        self.days = []
        self.families = []
        self.count = len(updates)
        if updates == []:
            self.table = None
        else:
            self.table = np.zeros((len(updates),
                                   parameters.period + parameters.padding))
        days_dict = set()
        for idx, update in enumerate(updates):
            self.families.append(update[1])
            days_dict |= {update[1], update[2]}
            self.table[idx][update[1]] += parameters.family_sizes[update[0]]
            self.table[idx][update[2]] += -parameters.family_sizes[update[0]]
        self.days = list(days_dict)
        self.occupancy_update = OccupancyUpdate(parameters)
        self.occupancy_update.days = self.days
        self.occupancy_update.count = self.count
        self.occupancy_update.table = \
            np.sum(self.table * parameters.family_sizes.reshape((-1,1)),
                   axis=0)


class Schedule:

    def __init__(self, schedule_parameters, init_schedule=None):
        self.parameters = schedule_parameters

        self.occupancy = np.zeros((2, schedule_parameters.period +
                                   schedule_parameters.padding))
        self.schedule = np.zeros((schedule_parameters.families,
                                  schedule_parameters.period +
                                  schedule_parameters.padding))
        self.schedule_cost = 0
        self.valid = False

        if init_schedule is not None:
            print("heeere")
            self.schedule = np.copy(init_schedule)
            self.update()
        return

    def _valid(self):
        return all([np.sum(self.schedule) == self.parameters.families,
                    (self.occupancy >= self.parameters.min_occupancy).all(),
                    (self.occupancy <= self.parameters.max_occupancy).all()])

    def _update_occupancy(self, update=None):
        occupancy = np.copy(self.occupancy)
        if update is not None:
            if update.count > 0:
                occupancy[0][update.days] += \
                    np.sum(update.table[:, update.days], axis=0)

                indices = set(update.days) | \
                          {day - 1 for day in update.days}
        else:
            occupancy[0] = np.sum(
                self.schedule * self.parameters.family_sizes.reshape(5000, 1),
                axis=0)
            indices = range(1, self.parameters.period + 1)

        for day in indices:
            occupancy[1][day] = \
                abs(occupancy[0][day] - occupancy[0][day + 1])

        return occupancy

    def _update_cost(self, update=None):
        if update is not None:
            if update.count > 0:
                ap_cost = self.parameters.accounting_penalty(
                    self.occupancy[0], self.occupancy[1])
                pref_cost = self.parameters.preference_cost(self.schedule)
                cost = ap_cost + pref_cost
            else:
                cost = self.schedule_cost
        else:
            cost = self.parameters.cost(self)


        return cost

    def update(self, update=None):
        if update is None:
            print("woot")
        occupancy = self._update_occupancy(update)
        cost = self._update_cost(update)

        self.occupancy = occupancy
        self.schedule_cost = cost
        self.valid = self._valid()

    def reset(self):
        self.occupancy.fill(0)
        self.schedule.fill(0)
        self.schedule_cost = 0
        self.valid = False


class Annealer:
    def __init__(self, parameters):
        self.parameters = parameters

        self.initial_temp = 0

        self.cooling_function = "linear"
        self.acceptance_probability = .5
        self.pc_nbhd = 1
        self.max_schedule_transitions = 10000
        self.equilibrium_transitions = 100

        self.cooling_factor = .9
        self.static_neighborhood = 10

        self.cooling_steps = 0
        self.cost_delta = 0

        self.min_temp = .5
        self.max_cooling_steps = 100
        self.max_static_steps = 10
        self.max_null_steps = 100

        self.temperature = 0
        self.null_steps = 0
        self.static_steps = 0
        self.cooling_steps = 0
        self.schedule_transitions = 0

        self.min_cost = None
        self.min_schedule = None

        #self.initial_temp = self.estimate_init_temp(.8, .1)
        self.temperature = self.initial_temp
        print("initial temp set to", self.initial_temp)
        #self.schedule = self.rand_init()
        return

    def _gen_transitions(self, sample_size, max_attempts):
        low_costs = []
        high_costs = []
        attempts = 0
        samples = 0

        while len(high_costs) < sample_size:
            schedule_0 = self.rand_init()
            cost_0 = schedule_0.schedule_cost
            schedule_1 = self.rand_init()
            cost_1 = schedule_1.schedule_cost
            if cost_1 > cost_0:
                low_costs.append(cost_0)
                high_costs.append(cost_1)
                samples += 1
            else:
                attempts += 1
            if samples == sample_size or attempts == max_attempts:
                break

        return np.array(low_costs), np.array(high_costs)

    def _ap_estimator(self, temperature, low_costs, high_costs):
        estimate = np.sum(np.exp(high_costs * (-1 / temperature))) / \
                   np.sum(np.exp(low_costs * (-1 / temperature)))
        return estimate

    def _ap_iterator(self, temperature, desired_tp, low_costs, high_costs, p):
        next_iter = temperature * math.pow((math.log(self._ap_estimator(
            temperature, low_costs, high_costs)) / math.log(desired_tp)),
                                           1 / p)
        return next_iter

    def estimate_init_temp(self, desired_tp, tolerance):
        samples = self._gen_transitions(50, 50)
        p = 1
        osc_window = [0, 0, 0]
        temp_init = 100000

        if len(samples[0]) > 0:
            low_costs, high_costs = samples
            temp_init = self._ap_iterator(temp_init, desired_tp, low_costs,
                                          high_costs, p)
            estimate = self._ap_estimator(temp_init, low_costs, high_costs)
            osc_window = [temp_init, temp_init, temp_init]
            osc_window_length = len(osc_window)
            print(temp_init)
            while abs(estimate - desired_tp) > tolerance:
                temp_init = temp_init * math.pow((math.log(
                    self._ap_estimator(temp_init, low_costs, high_costs)) /
                                                  math.log(desired_tp)), p)
                print(temp_init)
                for i in range(0, osc_window_length):
                    osc_window[i] = osc_window[(i - 1) % 3]
                osc_window[-1] = temp_init
                if (osc_window[2] - osc_window[1]) * (osc_window[1] -
                                                      osc_window[0]) < 0:
                    p *= 2
        print("pls work", temp_init)
        return temp_init

    def rand_init(self):
        new_schedule = Schedule(self.parameters)

        below_min_days = set(range(1, self.parameters.period + 1))
        below_threshold = below_min_days.copy()
        threshold = 10

        for family_id in self.parameters.family_ids:
            if len(below_threshold) == 0 and len(below_min_days) > 0:
                below_threshold = below_min_days.copy()

                threshold += 10

            if len(below_threshold) > 0 and len(below_min_days) > 0:
                day = random.choice(list(below_threshold))
                visitors = new_schedule.occupancy[0][day] + \
                           self.parameters.family_sizes[family_id]
                new_schedule.schedule[family_id][day ] = 1
                new_schedule.occupancy[0][day] = visitors

                if visitors >= self.parameters.min_occupancy:
                    below_min_days.remove(day)
                    below_threshold.remove(day)
                elif visitors >= threshold:
                    below_threshold.remove(day)

            elif len(below_min_days) == 0 and len(below_threshold) == 0:

                successful = False
                for _ in range(1, self.parameters.period + 1):
                    day = random.choice(range(1, self.parameters.period + 1))

                    visitors = new_schedule.occupancy[0][day] + \
                               self.parameters.family_sizes[family_id]
                    if visitors <= self.parameters.max_occupancy:
                        new_schedule.schedule[family_id][day] = 1
                        new_schedule.occupancy[0][day] = visitors
                        successful = True
                        break
                if not successful:
                    print("not successful")
                    new_schedule.reset()
        new_schedule.update()
        return new_schedule

    def neighbor(self, key, cost, tolerance):

        def _neighbor(move_list, cost):
            options = len(move_list)
            index = None
            if options == 1:
                index = 0
            elif options == 2:
                if abs(move_list[0][1] - cost) <= abs(move_list[1][1] - cost):
                    index = 0
                else:
                    index = 1
            elif options > 2:
                if cost == move_list[options // 2][1]:
                    return options // 2
                elif move_list[0][1] <= cost < move_list[options // 2][1]:
                    return _neighbor(move_list[0:options // 2], cost)

                else:
                    return options // 2 - 1 + \
                           _neighbor(move_list[options // 2 - 1:-1], cost)
            return index

        move_list = self.parameters.preference_delta[key]
        neighbor = None
        print("neighbors", key, move_list, cost)
        cost_bounds = [min(cost - tolerance, 0), cost + tolerance]
        lower_index = _neighbor(move_list, cost_bounds[0])
        upper_index = _neighbor(move_list, cost_bounds[1])

        if move_list[lower_index][1] < cost - tolerance:
            lower_index += 1

        if move_list[lower_index][1] > cost - tolerance:
            upper_index += -1

        if lower_index == upper_index:
            neighbor = move_list[lower_index]
        elif lower_index < upper_index:
            neighbor = move_list[random.randint(lower_index, upper_index)]

        return neighbor

    def propose_next(self):

        schedule_update = ScheduleUpdate(self.parameters, [])
        cost_delta = 0
        while True:
            days = random.sample(range(1, self.parameters.period + 1), 2)
            family_size = random.sample(self.parameters.family_size_pool, 1)[0]

            if self.schedule.occupancy[0][days[0]] + family_size > \
                    self.parameters.max_occupancy or \
                    self.schedule.occupancy[0][days[1]] - family_size < \
                    self.parameters.min_occupancy:
                break

            occupancy_update = OccupancyUpdate(self.parameters,
                                               [[family_size] + days])

            penalty_delta = self.parameters.ap_delta(self.schedule,
                                                     occupancy_update)
            print("penalty_delta", penalty_delta)
            swap = self.neighbor((days[0], days[1], family_size),
                                 penalty_delta, self.pc_nbhd)
            if swap is None:
                break

            elif self.schedule[days[0]][swap[0]] == 1 \
                    or self.schedule[days[1]][swap[0]] == 0:
                break
            else:
                schedule_update = ScheduleUpdate(self.parameters,
                                                 [[swap[0]] + days])
                cost_delta = penalty_delta + swap[1]
            break

        # TODO make sure cost_delta is computed in the right order.
        print(schedule_update.days, cost_delta)
        return schedule_update, cost_delta

    def accept_update(self, cost_delta):
        if cost_delta >= 0:
            return False
        else:
            ap = np.exp(-cost_delta / self.temperature)
            return np.random.choice([True, False], 1, p=[ap, 1 - ap])[0]

    def cool(self):
        self.cooling_steps += 1
        self.temperature = self.cooling_factor * self.temperature

        return self.temperature, self.cooling_steps

    def terminate(self):

        return any([self.temperature <= self.min_temp,
                    self.cooling_steps >= self.max_cooling_steps,
                    self.static_steps >= self.max_static_steps,
                    self.null_steps >= self.max_null_steps,
                    self.schedule_transitions >= self.max_schedule_transitions]
                   )

    def anneal(self):
        while self.terminate() is False:
            next_update, cost_delta = self.propose_next()
            if next_update.count > 0:
                if self.accept_update(cost_delta):
                    self.schedule.update(next_update)
                    print("schedule_cost", self.schedule.schedule_cost)
                    self.schedule_transitions += 1
                    if self.schedule.schedule_cost <= self.min_cost:
                        self.min_schedule = self.schedule

                    self.cost_delta += cost_delta
                    if self.cost_delta <= self.static_neighborhood:
                        self.static_steps += 1
                    else:
                        self.cost_delta = 0

                    if self.schedule_transitions % \
                            self.equilibrium_transitions == 0:
                        self.cool()
                        self.cooling_steps += 1
            else:
                self.null_steps += 1
            yield (next_update, cost_delta)

        print(f"The minimum cost achieved was {self.min_cost}")
        return


if __name__ == "__main__":
    parameters = ProblemSpec()
    print("yo")
    annealer = Annealer(parameters)
    for _ in range(0, 10):
        next(annealer.anneal())
        print(annealer.min_cost)

    # annealer.anneal()
