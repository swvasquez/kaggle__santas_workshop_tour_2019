import collections
import csv
import math
import pathlib
import random

import yaml

import numpy as np


class ScheduleParams:
    def __init__(self, path_string):

        self.families = 5000
        self.family_ids = np.arange(self.families)
        self.family_sizes = np.arange(self.families)
        self.family_sizes_pool = []

        self.days = 100
        self.min_occupancy = 125
        self.max_occupancy = 300

        self.preferences = 11
        self.pc_table = np.array([
            [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500],
            [0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434]])
        self.preference_costs = np.zeros((self.families, self.days +
                                          2 * self.padding))
        self.preference_delta = {}
        self.padding = 1
        self._parse_input(path_string)

    def _parse_input(self, path_string):
        path = pathlib.Path(path_string)

        with path.open(mode='r') as reader:
            csv_reader = csv.reader(reader)
            next(csv_reader)
            for row in csv_reader:
                family_id = int(row[0])
                family_size = int(row[11])
                self.family_sizes[family_id] = family_size
                self.preference_costs[family_id] = \
                    self.pc_table[0, 10] + family_size * self.pc_table[1, 10]
                for col in range(1, 11):
                    day = int(row[col])
                    self.preference_costs[family_id, day] = \
                        self.pc_table[0, col - 1] + \
                        family_size * self.pc_table[1, col - 1]

        self.family_sizes_pool += list(set(self.family_sizes))

    def _gen_pref_delta_dict(self):
        for day0 in range(1, self.days + 1):
            for day1 in range(1, self.days + 1):
                if day0 != day1:
                    for family_id in self.family_ids:
                        # TODO: Determine order of subtraction.
                        delta = self.preference_costs[family_id][day0] - \
                                self.preference_costs[family_id][day1]
                        key = (day0, day1, self.family_sizes[family_id])
                        if key in self.preference_delta:
                            self.preference_delta[key].append(
                                [family_id, delta])
                        else:
                            self.preference_delta[key] = [[family_id, delta]]

        for key in self.preference_delta:
            self.preference_delta[key].sort(key=lambda x: x[1])


    @staticmethod
    def accounting_penalty(daily_count, daily_shift):

        accounting_penalty = np.sum(
            .0025 * (daily_count - 125) *
            daily_count ** (1 / 2 + daily_shift / 50))

        return accounting_penalty

    def preference_cost(self, schedule):
        preference_cost = np.sum(self.preference_costs, schedule)
        return preference_cost

    def ap_delta(self, schedule, update):
        schedule_occupancy = np.copy(schedule.occupancy)

        schedule.occupancy[0] += np.sum(update, axis=0)

        indices = set(update.days) | {day - 1 for day in update.days}

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
                   update.table[update.days][update.families])
        return pref_delta

    def delta_cost(self, update):

        delta_cost = self.pc_delta(update) + self.ap_delta(update)

        return delta_cost

    # def ap_delta(self, days, family_size, occupancy):
    #     accounting_penalty = self.accounting_penalty
    #     window = 8
    #
    #     ap_delta = 0
    #     seg = np.zeros((2, window))
    #     lower = int(days[1] < days[0])
    #     gap = int((days[lower ^ 1] - days[lower]) == 1)
    #     shift_dict = {0: 4, 1: 1}
    #     sign_dict = {0: 1, 1: -1}
    #     shift = shift_dict[gap]
    #     # shift = -3 * gap + 4
    #     idx_dict = {0: [0, 1, 4, 5], 1: [0, 1, 2]}
    #
    #     seg[:, 0:3] = np.asarray(
    #         occupancy[:, days[lower] - 1: days[lower] + 2])
    #
    #     seg[:, shift:shift + 3] = np.asarray(
    #         occupancy[:, days[lower ^ 1] - 1: days[lower ^ 1] + 2])
    #
    #     ap_delta += -accounting_penalty(seg[0], seg[1])
    #
    #     seg[0][1] += sign_dict[lower] * family_size
    #     seg[0][1 + shift] += sign_dict[lower ^ 1] * family_size
    #
    #     for i in idx_dict[gap]:
    #         seg[1, i] = abs(seg[0][i] - seg[0][i + 1])
    #     ap_delta += accounting_penalty(seg[0], seg[1])
    #
    #     return ap_delta


class Update:
    def __init__(self, table, days, families):
        self.table = table
        self.days = days
        self.families = families


class Schedule:

    def __init__(self, schedule_parameters, input_schedule=None):
        self.parameters = schedule_parameters
        self.family_ids = schedule_parameters.family_ids
        self.family_sizes = schedule_parameters.family_sizes
        self.days = schedule_parameters.days
        self.min_occupancy = schedule_parameters.min_occupancy
        self.max_occupancy = schedule_parameters.max_occupancy
        self.accounting_penalty = schedule_parameters.accounting_penalty
        self.preference_costs = schedule_parameters.preference_costs
        self.accounting_penalty = schedule_parameters.accounting_penalty
        self.preference_cost = schedule_parameters.preference_cost

        self.occupancy = np.zeros((2, schedule_parameters.days), int)
        self.schedule = np.zeros((schedule_parameters.families,
                                  schedule_parameters.days), int)
        self.schedule_cost = 0

        if input_schedule:
            self.schedule = input_schedule
            self.update()
        return

    def reset(self):
        self.occupancy = np.zeros((2, schedule_parameters.days), int)
        self.schedule = np.zeros((schedule_parameters.families,
                                  schedule_parameters.days), int)
        self.schedule_cost = 0

    def _update_occupancy(self, update=None):

        if update is not None:
            self.occupancy[0][update.days] = \
                np.sum(self.schedule[:, update.days], axis=0)
            occupancy = np.copy(self.occupancy)
            indices = indices = set(update.days) | \
                                {day - 1 for day in update.days}
        else:
            occupancy = self.occupancy
            indices = range(1, self.days + 1)

        for day in indices:
            occupancy[1][day] = \
                abs(occupancy[0][day] - occupancy[0][day + 1])

        return occupancy

    def _update_cost(self, days=None, families=None, transitions=None):
        if any(arg is None for arg in [days, families, transitions]):
            ap_cost = self.accounting_penalty(self.occupancy[0],
                                              self.occupancy[1])
            pref_cost = self.preference_cost(self.schedule)
            self.schedule_cost = ap_cost + pref_cost
        else:
            self.schedule_cost += self.delta_cost(days, families, transitions)
        return

    def update(self, days=None, families=None, transitions=None):
        if any(arg is None for arg in [days, families, transitions]):
            self._update_occupancy()
            self._update_cost()
        else:
            self.schedule_cost += self.delta_cost(days, families, transitions)


class Annealer:
    def __init__(self, parameters, schedule):

        self.schedule = schedule
        self.days = parameters.days
        self.fs_pool = parameters.fs_pool
        self.max_occupancy = parameters.max_occupancy
        self.min_occupancy = parameters.min_occupancy
        self.ap_delta = parameters.ap_delta
        self.preference_delta = parameters.preference_delta
        self.family_ids = parameters.family_ids
        self.family_sizes = parameters.family_sizes

        self.initial_temp = 0
        self.min_temp = .5
        self.cooling_function = "linear"
        self.acceptance_probability = .5
        self.pc_nbhd = 1
        self.max_schedule_transitions = 1
        self.max_cooling_steps = 100
        self.max_static_steps = 10
        self.max_none_steps = 100
        self.cooling_factor = .9

        self.temperature = 0
        self.cooling_steps = 0
        self.cost_delta = 0
        self.static_steps = 0
        self.schedule_transitions = 0
        self.cost_delta_min = 0
        self.none_steps = 0

        self.min_cost = schedule.cost
        self.min_schedule = schedule

        return

    def _gen_transitions(self, steps, reset, terminate):
        low_costs = np.zeros(steps)
        high_costs = np.zeros(steps)
        attempts = 0
        samples = 0

        while len(high_costs) < steps:
            attempts = 0
            self.rand_init()
            cost_0 = self.schedule.cost
            cost_1 = cost_0
            samples += 1
            while cost_1 <= cost_0 and attempts < reset:
                self.rand_init()
                cost_1 = self.schedule.cost
                attempts += 1
            if samples < terminate:
                return None
        return low_costs, high_costs

    def _ap_estimator(self, temperature, low_costs, high_costs):
        estimate = np.sum(np.exp(high_costs * (-1 / temperature))) / \
                   np.sum(np.exp(low_costs * (-1 / temperature)))

        return estimate

    def estimate_init_temp(self, desired_tp, tolerance):
        samples = self._gen_transitions(100, 10, 10)
        p = 1
        osc_window = [0, 0, 0]

        if samples is not None:
            low_costs, high_costs = samples
            temp_init = -np.sum(high_costs - low_costs) / len(high_costs) * \
                        math.log(desired_tp)
            estimate = self._ap_estimator(temp_init, low_costs, high_costs)
            osc_window = [temp_init, temp_init, temp_init]
            osc_window_length = len(osc_window)

            while abs(estimate - desired_tp) < tolerance:
                temp_init = temp_init * math.pow((math.log(self._ap_estimator(
                    low_costs, high_costs)) / math.log(desired_tp)), p)
                for i in range(0, osc_window_length):
                    osc_window[i] = osc_window[(i - 1) % 3]
                osc_window[-1] = temp_init
                if (osc_window[2] - osc_window[1]) * (osc_window[1] -
                                                      osc_window[0]) < 0:
                    p *= 2
            self.initial_temp = temp_init
            return temp_init
        else:
            return None

    def rand_init(self):
        below_min_days = set(range(0, self.days))
        below_threshold = below_min_days.copy()
        threshold = 10
        for family_id in self.family_ids:
            if len(below_threshold) == 0 and len(below_min_days) > 0:
                below_threshold = below_min_days.copy()

                threshold += 10

            if len(below_threshold) > 0 and len(below_min_days) > 0:
                day = random.choice(list(below_threshold))
                visitors = self.occupancy[0][day] + \
                           self.family_sizes[family_id]
                self.schedule[family_id][day - 1] = 1
                self.occupancy[0][day] = visitors

                if visitors >= self.min_occupancy:
                    below_min_days.remove(day)
                    below_threshold.remove(day)
                elif visitors >= threshold:
                    below_threshold.remove(day)

            elif len(below_min_days) == 0 and len(below_threshold) == 0:
                successful = False
                for _ in range(0, self.days):
                    day = random.choice(range(0, self.days))

                    visitors = self.occupancy[0][day] + \
                               self.family_sizes[family_id]
                    if visitors <= self.max_occupancy:
                        self.schedule[family_id][day] = 1
                        self.occupancy[0][day] = visitors
                        successful = True
                        break
                if not successful:
                    self.occupancy = np.zeros((2, self.days), int)
                    return None

        for day in range(1, self.days - 1):
            self.occupancy[1][day] = \
                abs(self.occupancy[0][day] - self.occupancy[0][day + 1])

        self.schedule.update()

        return

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

        move_list = self.preference_delta[key]
        neighbor = None

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

        update = None
        while True:
            days = random.sample(range(0, self.days), 2)
            family_size = random.sample(self.fs_pool, 1)

            if self.schedule.occupancy[days[0]] + family_size > \
                    self.max_occupancy or \
                    self.schedule.occupancy[days[1]] - family_size < \
                    self.min_occupancy:
                continue
            if self.schedule[days[0]][swap[0]] == 1 \
                    or self.schedule[days[1]][swap[0]] == 0:
                continue

            penalty_delta = self.ap_delta(days, family_size)
            potential_swaps = self.preference_delta[(days[0], days[1],
                                                     family_size)]
            swap = self.schedule.neighbor(potential_swaps, penalty_delta,
                                          self.pc_nbhd)
            if swap is None:
                continue
            break
        # TODO make sure cost_delta is computed in the right order.
        cost_delta = penalty_delta + swap[1]
        family_id = swap[0]
        update = (days, family_id, cost_delta)

        return update

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
                    self.none_steps >= self.max_none_steps,
                    self.schedule_transitions >=
                    self.max_schedule_transitions])

    def anneal(self):
        while self.terminate() is False:
            next_state = self.propose_next()
            if next_state is not None:
                family_id, days, cost_delta = next_state
                if self.accept_update(cost_delta):
                    __, _, schedule_cost, cost_delta = \
                        self.schedule.update(days,
                                             [family_id, family_id], [1, 0])
                    if schedule_cost <= self.min_cost:
                        self.min_schedule = self.schedule
                    self.schedule_transitions += 1
                    self.cost_delta += cost_delta
                    if self.cost_delta < self.cost_delta_min:
                        self.static_steps += 1
                    if self.schedule_transitions % \
                            self.max_schedule_transitions == 0:
                        self.cool()
            else:
                self.none_steps += 1
            if self.terminate():
                break
        print(f"The minimum cost achieved was {self.schedule.schedule_cost}")
        return


if __name__ == "__main__":
    config_path = pathlib.Path('../../config.yaml')
    with config_path.open(mode='r') as config:
        docs = yaml.load_all(config, Loader=yaml.FullLoader)

        for doc in docs:

            for k, v in doc.items():
                print(k, "->", v)
    #
    # file_path = "/home/scott/Projects/kaggle__santa's_workshop_tour_2019[" \
    #             "python]/data/family_data.csv"
    # schedule_parameters = ScheduleParams(file_path)
    # print("family_size", max(schedule_parameters.family_sizes))
    # schedule = Schedule(schedule_parameters)
