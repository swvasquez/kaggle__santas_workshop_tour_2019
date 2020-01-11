import csv
import curses
import json
import math
import random
import signal
import sys
import time
import yaml

import numpy as np
import redis

from simulated_annealing import project_paths


class ProblemSpec:

    def __init__(self, display):

        self.display = display

        paths = project_paths()
        root = paths['root']
        config_path = paths['config']

        self.display.display_progress(status='Loading config file.')

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
        self.padding = 2
        self.preference_costs = np.zeros((self.families, self.period +
                                          self.padding))

        self.output_dir = root / data['output_dir']

        self.display.display_progress(status='Reading family preference CSV.')
        fsp, fs, pc = self._parse_family_preferences_csv(root / data[
            'preference_csv'])
        self.family_size_pool = fsp
        self.family_sizes = fs
        self.preference_costs = pc

        self.display.display_progress(status='Connecting to Redis database.')
        self.redis_cxn = redis.Redis(
            host='localhost',
            port=6379,
            charset="utf-8",
            decode_responses=True
        )

        if self.redis_cxn.dbsize() != 69300:
            self.display.display_progress(status='Creating Redis ' \
                                                 'database.')
            self.redis_cxn.flushdb()
            self._gen_pref_delta_dict()

        self.display.display_progress(status='')

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

        return family_size_pool, family_sizes, preference_costs

    def _gen_pref_delta_dict(self):

        for size in self.family_size_pool:
            preference_delta = {}
            for day0 in range(1, self.period + 1):
                for day1 in range(1, self.period + 1):
                    if day0 != day1:
                        for family_id in self.family_ids:
                            if int(self.family_sizes[family_id]) == int(size):
                                delta = self.preference_costs[family_id][
                                            day0] - \
                                        self.preference_costs[family_id][day1]
                                key = (
                                    day0, day1,
                                    int(self.family_sizes[family_id]))

                                if key in preference_delta:
                                    if delta in preference_delta[key]:
                                        preference_delta[key][delta].append(
                                            int(family_id))
                                    else:
                                        preference_delta[key][delta] = [
                                            int(family_id)]
                                else:
                                    preference_delta[key] = {delta: [int(
                                        family_id)]}

            for key in preference_delta:
                data_list = [[delta] + preference_delta[key][delta] for
                             delta in preference_delta[key]]
                data_list.sort(key=lambda x: x[0])
                self.redis_cxn.set(json.dumps(key), json.dumps(
                    data_list.copy()))

            return

    def accounting_penalty(self, daily_occupancy, daily_shift):
        accounting_penalty = np.sum(
            .0025 * (daily_occupancy - 125) *
            daily_occupancy ** (1 / 2 + daily_shift / 50))
        return accounting_penalty

    def preference_cost(self, schedule):
        preference_cost = np.sum(self.preference_costs * schedule)
        return preference_cost

    def cost(self, schedule, occupancy):
        ap = self.accounting_penalty(occupancy[0], occupancy[1])
        pc = self.preference_cost(schedule)
        return ap + pc

    def ap_delta(self, occupancy, occupancy_update):
        occupancy_copy = np.copy(occupancy)
        occupancy_copy[0] += occupancy_update.table

        indices = set(occupancy_update.days) | \
                  {day - 1 for day in occupancy_update.days}

        for idx in indices:
            occupancy_copy[1][idx] = abs(occupancy_copy[0][idx + 1]
                                         - occupancy_copy[0][idx])

        new_cost = self.accounting_penalty(
            occupancy_copy[[0], list(indices)],
            occupancy_copy[[1], list(indices)])

        old_cost = self.accounting_penalty(occupancy[[0], list(indices)],
                                           occupancy[[1], list(indices)])

        ap_delta = new_cost - old_cost

        return ap_delta

    def pc_delta(self, schedule_update):

        pref_delta = np.sum(self.preference_costs[schedule_update.families]
                            * schedule_update.table)
        return pref_delta

    def delta_cost(self, occupancy, schedule_update):

        delta_cost = self.pc_delta(schedule_update) + \
                     self.ap_delta(occupancy, schedule_update.occupancy_update)

        return delta_cost


class OccupancyUpdate:
    def __init__(self, parameters, updates=None):
        if updates is None:
            updates = []
        self.days = []
        self.count = len(updates)
        self.table = np.zeros(parameters.period + parameters.padding)

        days = set()
        for update in updates:
            days |= {update[1], update[2]}
            self.table[update[1]] += update[0]
            self.table[update[2]] += -update[0]
        self.days = list(days)


class ScheduleUpdate:
    def __init__(self, parameters, updates=None):
        if updates is None:
            updates = []
        self.days = []
        self.families = []
        self.count = len(updates)

        self.occupancy_update = OccupancyUpdate(parameters)
        self.occupancy_update.count = self.count

        if updates == []:
            self.table = np.zeros(parameters.period + parameters.padding)
        else:
            self.table = np.zeros((len(updates),
                                   parameters.period + parameters.padding))
            days = set()
            for idx, update in enumerate(updates):
                self.families.append(update[0])
                days |= {update[1], update[2]}
                self.table[idx][update[1]] += 1
                self.table[idx][update[2]] += -1
            self.days = list(days)

            self.occupancy_update.table = \
                np.sum(self.table * parameters.family_sizes.reshape((-1, 1))[
                                    self.families, :], axis=0)

        self.occupancy_update.days = self.days


class Schedule:

    def __init__(self, parameters, init_schedule=None):
        self.parameters = parameters

        self.occupancy = np.zeros((2, parameters.period + parameters.padding))
        self.schedule = np.zeros((parameters.families, parameters.period +
                                  parameters.padding))
        self.schedule_cost = 0
        self.valid = False

        if init_schedule is not None:
            self.schedule = np.copy(init_schedule)
            self.update()
        return

    def _valid(self):
        return all([np.sum(self.schedule) == self.parameters.families,
                    (self.occupancy >= self.parameters.min_occupancy).all(),
                    (self.occupancy <= self.parameters.max_occupancy).all()])

    def _update_schedule(self, update=None):
        table = np.copy(self.schedule)
        if update is not None and update.count > 0:
            table[update.families, update.days] += update.table[:,
                                                   update.days][0]
        return table

    def _update_occupancy(self, update=None):
        occupancy = np.copy(self.occupancy)
        if update is not None:
            if update.count > 0:
                occupancy[0][update.days] += update.table[update.days]
                indices = set(update.days) | \
                          {day - 1 for day in update.days}
            else:
                return occupancy
        else:
            occupancy[0] = np.sum(
                self.schedule * self.parameters.family_sizes.reshape(-1, 1),
                axis=0)
            indices = range(1, self.parameters.period + 1)

        for day in indices:
            occupancy[1][day] = \
                abs(occupancy[0][day] - occupancy[0][day + 1])
        occupancy[1][-2] = 0
        return occupancy

    def _update_cost(self, update=None):
        current_cost = self.schedule_cost
        if update is not None:
            if update.count > 0:
                delta_cost = self.parameters.delta_cost(self.occupancy, update)

                cost = delta_cost + current_cost
            else:
                cost = current_cost
        else:
            cost = self.parameters.cost(self.schedule, self.occupancy)

        return cost

    def update(self, update=None, delta_cost=None):
        schedule = self._update_schedule(update)
        if update is None:
            occupancy = self._update_occupancy()
        else:
            occupancy = self._update_occupancy(update.occupancy_update)

        self.schedule = schedule
        self.occupancy = occupancy
        if delta_cost is None:
            self.schedule_cost = self._update_cost(update)
        else:
            self.schedule_cost += delta_cost
        self.valid = self._valid()

    def reset(self):
        self.occupancy.fill(0)
        self.schedule.fill(0)
        self.schedule_cost = 0
        self.valid = False


class Annealer:
    def __init__(self, parameters, display):

        self.parameters = parameters
        self.display = display

        paths = project_paths()
        config_path = paths['config']
        root_path = paths['root']
        self.display.display_progress(status='Loading annealer settings.')

        with config_path.open(mode='r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.display.display_progress(status='')
        annealer_specs = config['annealer']

        self.snapshots_dir = root_path / config['data']['snapshot_dir']
        self.initial_temp = 0

        self.lower_nbhd_bound = int(annealer_specs['lower_nbhd_bound'])
        self.upper_nbhd_bound = int(annealer_specs['upper_nbhd_bound'])
        self.static_nbhd = int(annealer_specs['static_nbhd'])

        self.max_static_transitions = int(annealer_specs[
                                              'max_static_transitions'])
        self.max_null_steps = int(annealer_specs['max_null_steps'])
        self.equilibrium_transitions = \
            int(annealer_specs['equilibrium_transitions'])

        self.cooling_factor = float(annealer_specs['cooling_factor'])
        self.min_temperature = float(annealer_specs['min_temperature'])

        self.snapshot_delta = int(annealer_specs['snapshot_delta'])

        self.cooling_steps = 0
        self.cost_delta = 0

        self.temperature = 0
        self.null_steps = 0
        self.static_steps = 0
        self.cooling_steps = 0
        self.positive_transitions = 0
        self.proposed_transitions = 0
        self.accepted_transitions = 0
        self.steps = 0

        self.min_cost = None
        self.min_schedule = None

        self.initial_temp = 33000
        self.temperature = self.initial_temp

        self.schedule = self.rand_init()
        self.min_cost = self.schedule.schedule_cost
        self.all_time_min = self.schedule.schedule_cost

        for file_path in self.snapshots_dir.iterdir():
            file_cost = int(file_path.stem)
            if self.all_time_min is None:
                self.all_time_min = file_cost
            else:
                self.all_time_min = min(self.all_time_min, file_cost)

        display_dict = {
            "temperature": self.initial_temp,
            "temp_init": self.initial_temp,
            "cost": self.schedule.schedule_cost,
            "cost_init": self.schedule.schedule_cost
        }
        self.display.display_progress(**display_dict)

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
        samples = self._gen_transitions(1000, 50)
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
            while abs(estimate - desired_tp) > tolerance:
                temp_init = temp_init * math.pow((math.log(
                    self._ap_estimator(temp_init, low_costs, high_costs)) /
                                                  math.log(desired_tp)), p)

                for i in range(0, osc_window_length):
                    osc_window[i] = osc_window[(i - 1) % 3]
                osc_window[-1] = temp_init
                if (osc_window[2] - osc_window[1]) * (osc_window[1] -
                                                      osc_window[0]) < 0:
                    p *= 2
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
                new_schedule.schedule[family_id][day] = 1
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
                    new_schedule.reset()
        new_schedule.update()
        return new_schedule

    def _suggest_neighbor(self, key, cost, tol_less, tol_more):

        def _neighbor(move_list, cost):
            options = len(move_list)
            index = None
            if options == 1:
                index = 0
            elif options == 2:
                if abs(move_list[0][0] - cost) <= abs(move_list[1][0] - cost):
                    index = 0
                else:
                    index = 1
            elif options > 2:
                if cost == move_list[options // 2][0]:
                    return options // 2
                elif cost < move_list[options // 2][0]:
                    return _neighbor(move_list[0:options // 2], cost)

                else:
                    return options // 2 + \
                           _neighbor(move_list[options // 2:], cost)
            return index

        move_list = json.loads(self.parameters.redis_cxn.get(json.dumps(key)))

        neighbor = None

        lower_index = _neighbor(move_list, cost - tol_less)
        upper_index = _neighbor(move_list, cost + tol_more)

        if move_list[lower_index][0] < cost - tol_less:
            lower_index += 1

        if move_list[upper_index][0] > cost + tol_more:
            upper_index += -1

        if lower_index > upper_index:
            return None
        else:

            samples = []
            cost_dict = {}

            for idx in range(lower_index, upper_index + 1):
                samples += move_list[idx][1:]
                for family_id in move_list[idx][1:]:
                    cost_dict[family_id] = move_list[idx][0]

            neighbor = random.choice(samples)
            cost_delta = cost_dict[neighbor]

            return [neighbor, cost_delta]

    def _propose_next(self):

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

            penalty_delta = self.parameters.ap_delta(self.schedule.occupancy,
                                                     occupancy_update)

            swap = self._suggest_neighbor((days[0], days[1], int(family_size)),
                                          -penalty_delta,
                                          self.lower_nbhd_bound,
                                          self.upper_nbhd_bound)

            if swap is None:

                break


            elif self.schedule.schedule[swap[0]][days[0]] == 1 or \
                    self.schedule.schedule[swap[0]][days[1]] == 0:

                break
            else:
                schedule_update = ScheduleUpdate(self.parameters,
                                                 [[swap[0]] + days])
                cost_delta = penalty_delta + swap[1]

            break

        return schedule_update, cost_delta

    def _accept_update(self, cost_delta):
        if cost_delta <= 0:
            return True
        else:
            ap = np.exp(-cost_delta / self.temperature)
            accept = np.random.choice([True, False], 1, p=[ap, 1 - ap])[0]
            return accept

    def _cool(self):
        self.temperature = self.cooling_factor * self.temperature

        return self.temperature, self.cooling_steps

    def _terminate(self):
        termination = False
        termination_dict = {
            "Minimum temp reached. ": self.temperature <= self.min_temperature,
            "Cost not changing. ": self.static_steps >=
                                   self.max_static_transitions,
            "Not generating transitions.": self.null_steps >=
                                           self.max_null_steps
        }
        status = ''
        for key in termination_dict:
            if termination_dict[key]:
                status += key
        if status != '':
            termination = True
            self.display.display_progress(status=status)
        return termination

    def _save(self):
        file_path = self.snapshots_dir / (str(int(
            self.min_cost)) + '.npy')
        np.save(file_path.resolve().as_posix(),
                self.schedule.schedule)
        pass


    def anneal(self):
        while self._terminate() is False:
            next_update, cost_delta = self._propose_next()
            if next_update.count > 0:
                self.proposed_transitions += 1
                if self._accept_update(cost_delta):
                    self.schedule.update(next_update, cost_delta)
                    self.accepted_transitions += 1
                    if self.schedule.schedule_cost < self.min_cost:
                        self.min_cost = self.schedule.schedule_cost
                        if self.min_cost + self.snapshot_delta < self.all_time_min:
                            self._save()
                            self.all_time_min = self.min_cost

                    self.cost_delta += cost_delta
                    if self.cost_delta <= self.static_nbhd:
                        self.static_steps += 1
                    else:
                        self.cost_delta = 0

                    if self.accepted_transitions % \
                            self.equilibrium_transitions == 0:
                        self._cool()
                        self.cooling_steps += 1

                    display_dict = {
                        "status": 'Running...',
                        "temperature": self.temperature,
                        "temp_init": self.initial_temp,
                        "cost": self.schedule.schedule_cost,
                        "step": self.steps,
                        "min": self.min_cost,
                        "all_time_min": self.all_time_min,
                        "transitions": self.accepted_transitions,
                        "cd_average": self.display.cd_average
                                      * (
                                              self.display.transitions / self.proposed_transitions)
                                      + cost_delta / self.proposed_transitions
                    }
                    self.display.display_progress(**display_dict)
            else:
                self.null_steps += 1
            self.steps += 1
            yield next_update, cost_delta


class Display:
    def __init__(self, stdscr, **attrs):

        self.cost = 0
        self.temperature = 0
        self.temp_init = 0
        self.cost_init = 0
        self.transitions = 1
        self.positive_transitions = 0
        self.step = 0
        self.curses = stdscr
        self.min = 0
        self.all_time_min = 0
        self.status = ''
        self.cd_average = 0

        for key, val in attrs.items():
            if key in vars(self):
                setattr(self, key, val)

    def display_progress(self, **attrs):
        self.curses.clear()
        for key, val in attrs.items():
            if key in vars(self):
                setattr(self, key, val)

        self.curses.addstr(0, 0, f"Temp:{self.temperature} / {self.temp_init}")
        self.curses.addstr(1, 0, f"Cost:{self.cost} / {self.cost_init}")
        self.curses.addstr(2, 0, f"Min:{self.min} / {self.all_time_min}")

        self.curses.addstr(3, 0, f"Average move cost: {self.cd_average}")
        self.curses.addstr(4, 0, f"Step:{self.step}")
        self.curses.addstr(6, 0, self.status)
        self.curses.refresh()



def npy_to_csv(npy_path, output_dir):
    data = np.load(npy_path)

    days = np.where(data==1)[1]
    print(days)
    output_path = output_dir / f"{npy_path.stem}.csv"
    with output_path.open(mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["family_id", "assigned_day"])
        for row in range(0,data.shape[0]):
            writer.writerow([row,days[row]])



def signal_handler(annealer, start_time):
    def _signal_handler(signum, frame):
        curses.endwin()
        end_time = time.perf_counter()
        delta = end_time - start_time
        print(f"\n\r{annealer.proposed_transitions} transitions, {delta} "
              f"seconds")
        print(f"\r{annealer.proposed_transitions / delta} t/s")

        npy_path = annealer.snapshots_dir / f"{int(annealer.all_time_min)}.npy"
        if npy_path.is_file():
            npy_to_csv(npy_path,annealer.snapshots_dir)
        sys.exit(0)

    return _signal_handler


if __name__ == "__main__":

    stdscr = curses.initscr()
    curses.noecho()
    try:
        display = Display(stdscr)

        parameters = ProblemSpec(display)
        annealer = Annealer(parameters, display)
        start_time = time.perf_counter()
        signal.signal(signal.SIGINT, signal_handler(annealer, start_time))
        while True:
            next(annealer.anneal())
    except Exception as e:
        curses.endwin()
        raise e
