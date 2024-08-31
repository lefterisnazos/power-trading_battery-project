import cvxpy as cp
import numpy as np
import time
import pandas as pd
import random

# HAVE TO DO: pip install cvxopt

# original allocation function, only for testing purposes

def allocate_prices(price_dict, init_state, end_state, alpha):
    """
    instead of having bid and ask dictionaries, we just have a prices dictionaries, so we buy and sell on the same price.
    """
    n = len(price_dict)
    price_array = np.fromiter(price_dict.values(), dtype=float)
    x_buy = cp.Variable(n, boolean=True)
    x_sell = cp.Variable(n, boolean=True)
    objective = cp.Maximize(alpha * (price_array @ (x_sell - x_buy)) - (1-alpha) * cp.sum(x_buy + x_sell))

    constraints = [x_buy + x_sell <= 1,  # buy and sell can't occur at the same time
                   cp.cumsum(x_buy - x_sell)[n - 1] == end_state - init_state,  # cumulative sum of all the actions has to be zero (initial and last state have to be the same)
                   cp.cumsum(x_buy - x_sell) >= 0 - init_state,  # initial state + cumulative sum of the actions should always be larger than 0
                   cp.cumsum(x_buy - x_sell) <= 4 - init_state]  # initial state + cumulative sum of the actions should always be smaller than 4

    prob = cp.Problem(objective, constraints)
    prob.solve()

    max_profit = price_array @ (x_sell - x_buy)
    x_opt = x_buy.value - x_sell.value
    num_actions = np.sum(x_sell.value, dtype=int)

    return x_opt, max_profit, num_actions


def allocate_bid_ask(ask_dict, bid_dict, init_state, end_state, alpha):
    n = len(ask_dict)
    ask_array = np.fromiter(ask_dict.values(), dtype=float)
    bid_array = np.fromiter(bid_dict.values(), dtype=float)

    # Construct a CVXPY problem
    x_ask = cp.Variable(n, boolean=True)
    x_bid = cp.Variable(n, boolean=True)
    objective = cp.Maximize(alpha * (bid_array @ x_bid - ask_array @ x_ask) - (1 - alpha) * cp.sum(x_ask + x_bid))  # spend/buy on the ask, earn/sell on the bid
    constraints = [x_ask + x_bid <= 1,  # buy and sell can't occur at the same time
                   cp.cumsum(x_ask - x_bid)[n - 1] == end_state - init_state,  # cumulative sum of all the actions has to be zero (initial and last state have to be the same)
                   cp.cumsum(x_ask - x_bid) >= 0 - init_state,  # initial state + cumulative sum of the actions should always be larger than 0
                   cp.cumsum(x_ask - x_bid) <= 4 - init_state]  # initial state + cumulative sum of the actions should always be smaller than 4

    prob = cp.Problem(objective, constraints)
    # prob.solve(verbose=True)
    prob.solve(verbose=False)

    # extract the values
    max_profit = bid_array @ x_bid.value - ask_array @ x_ask.value
    x_opt = x_ask.value - x_bid.value
    num_actions = np.sum(x_ask.value, dtype=int)
    return x_opt, max_profit, num_actions


# close the bids if possible (if there exists any suballocation which brings profit)
def close_bids(old_ask, old_bid, ask, bid, old_alloc):
    n_old = len(old_ask)
    old_ask_array, old_bid_array = np.fromiter(old_ask.values(), dtype=float), np.fromiter(old_bid.values(), dtype=float)
    ask_array, bid_array = np.fromiter(ask.values(), dtype=float), np.fromiter(bid.values(), dtype=float)
    ask_array, bid_array = ask_array[:n_old], bid_array[:n_old]

    old_x_bid = np.array([int(item) for item in old_alloc < 0])
    old_x_ask = np.array([int(item) for item in old_alloc > 0])

    # Construct a CVXPY problem
    x_ask = cp.Variable(n_old, boolean=True)
    x_bid = cp.Variable(n_old, boolean=True)
    objective = cp.Maximize(0)  # solve feasibilty problem
    constraints = [cp.sum(x_ask - x_bid) == 0, old_x_bid - x_ask >= 0, old_x_ask - x_bid >= 0,
                   (bid_array @ x_bid - old_ask_array @ old_x_ask) + (old_bid_array @ old_x_bid - ask_array @ x_ask) >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x_ask.value, x_bid.value


# from old allocation remove the keys which are "expired"
def trim_old_allocation(old_ask, old_bid, new_ask, old_alloc):
    n_old = len(old_ask)  # number of old bids and asks
    # we assume the timesteps are corectly ordered and that bid and ask dictionaries have the same keys!!!
    n_diff = 0
    old_ask_trim, old_bid_trim = {}, {}
    for key in old_ask:
        if key not in new_ask.keys():
            n_diff += 1
        else:
            old_ask_trim[key] = old_ask[key]
            old_bid_trim[key] = old_bid[key]
    # find old allocation and extract allocations only for keys which exists in new bids/asks
    old_alloc_trim = old_alloc[n_diff:n_old]

    return old_ask_trim, old_bid_trim, old_alloc_trim


def allocate_bid_ask_new(ask, bid, old_ask, old_bid, old_alloc, init_state, end_state, alpha):
    n, n_old = len(ask), len(old_ask)
    ask_array, bid_array = np.fromiter(ask.values(), dtype=float), np.fromiter(bid.values(), dtype=float)
    old_ask_array, old_bid_array = np.fromiter(old_ask.values(), dtype=float), np.fromiter(old_bid.values(), dtype=float)

    old_x_bid = np.array([int(item) for item in old_alloc < 0])  # extract old bid allocation
    old_x_ask = np.array([int(item) for item in old_alloc > 0])  # extract old ask allocation

    # Construct a CVXPY problem
    x_ask = cp.Variable(n, boolean=True)
    x_bid = cp.Variable(n, boolean=True)

    bought_before_sold_now = cp.multiply(x_bid[0:n_old], old_x_ask)  # logical vector for the positions bought before and sold now
    sold_before_bought_now = cp.multiply(x_ask[0:n_old], old_x_bid)  # logical vector for the positions sold before and bought now
    bought_before_na_now = cp.multiply(1 - x_bid[0:n_old] - x_ask[0:n_old], old_x_ask)  # logical vector for the positions bought before and non active now
    sold_before_na_now = cp.multiply(1 - x_ask[0:n_old] - x_bid[0:n_old], old_x_bid)  # logical vector for the positions sold before and non active now
    bought_before_bought_now = cp.multiply(x_ask[0:n_old], old_x_ask)  # logical vector for the positions bought before and bought now
    sold_before_sold_now = cp.multiply(x_bid[0:n_old], old_x_bid)  # logical vector for the positions sold before and sold now
    objective = cp.Maximize(alpha * (bid_array @ x_bid - ask_array @ x_ask +  # nominal profit
                                     (old_bid_array - ask_array) @ sold_before_na_now +  # -1 -> 0
                                     (bid_array - old_ask_array) @ bought_before_na_now +  # 1 -> 0
                                     (old_bid_array - ask_array) @ sold_before_bought_now +  # -1 -> 1
                                     (bid_array - old_ask_array) @ bought_before_sold_now +  # 1 -> -1
                                     (ask_array - old_ask_array) @ bought_before_bought_now +  # 1 -> 1
                                     (old_bid_array - bid_array) @ sold_before_sold_now) - (1 - alpha) * cp.sum(x_ask + x_bid))  # -1 -> -1

    constraints = [x_ask + x_bid <= 1,  # buy and sell can't occur at the same time
                   cp.cumsum(x_ask - x_bid)[n - 1] == end_state - init_state,  # cumulative sum of all the actions has to be zero (initial and last state have to be the same)
                   cp.cumsum(x_ask - x_bid) >= 0 - init_state,  # initial state + cumulative sum of the actions should always be larger than 0
                   cp.cumsum(x_ask - x_bid) <= 4 - init_state]  # initial state + cumulative sum of the actions should always be smaller than 4

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # extract the values
    max_profit = objective.value
    x_opt = x_ask.value - x_bid.value
    num_actions = np.sum(x_ask.value, dtype=int)

    return x_opt, max_profit, num_actions


def get_random_ask_bid(length=32):
    n_bids = length  # number of asks/bids
    old_ask, old_bid = {}, {}
    keys = np.array(random.sample(range(1, 100), n_bids))
    for i in range(0, n_bids):
        rand_bid = round(450 * np.random.rand(), 2)
        old_bid[keys[i]] = rand_bid
        old_ask[keys[i]] = round(rand_bid + (500 - rand_bid) * np.random.rand(), 2)

    new_ask, new_bid = dict.fromkeys(old_ask), dict.fromkeys(old_bid)
    for i in range(0, n_bids):
        rand_bid = round(450 * np.random.rand(), 2)
        new_bid[keys[i]] = rand_bid
        new_ask[keys[i]] = round(rand_bid + (500 - rand_bid) * np.random.rand(), 2)

    return old_bid, old_ask, new_bid, new_ask


def get_initial_allocation(ask_dict, bid_dict, init_state=4, end_state=4, alpha=1):
    old_ask = ask_dict
    old_bid = bid_dict

    initial_allocation, initial_profit, number_of_actions = allocate_bid_ask(old_ask, old_bid, init_state, end_state, 1)

    return initial_allocation, np.round(initial_profit,2), number_of_actions


def get_new_allocation(ask, bid, old_ask, old_bid, old_allocation, init_state, end_state, alpha=1):
    old_ask, old_bid, old_allocation = trim_old_allocation(old_ask, old_bid, ask, old_allocation)  # remove the keys that have expired
    new_allocation, profit, number_of_actions = allocate_bid_ask_new(ask, bid, old_ask, old_bid, old_allocation, init_state, end_state, alpha)

    # profit  = allocation_profit + cost_profit
    allocation_profit, cost_profit = allocation_part_cost_part(old_ask, old_bid, ask, bid,old_allocation, new_allocation)
    next_old_ask, next_old_bid = get_true_current_bid_ask(old_ask, old_bid, ask, bid, old_allocation, new_allocation)

    return [[new_allocation, profit, number_of_actions]], [allocation_profit, cost_profit], [next_old_ask, next_old_bid, old_allocation]


def get_initial_allocation2(ask_dict, bid_dict, init_state=4, end_state=4, alpha=1):
    old_ask = ask_dict
    old_bid = bid_dict

    new_allocation, profit, number_of_actions = None, None, None
    n_alpha = 100  # choose the appropriate number of alpha values
    alpha = np.linspace(0, 1.0, num=n_alpha)
    new_alloc_list, new_profit_list, num_actions_list, fractions_list = [], [], [], []
    info_of_best = None
    best_fraction = -0.0001
    cost_factor_for1mwtrade = 0
    start = time.time()
    for i in range(0, n_alpha):
        new_allocation, profit, number_of_actions = allocate_bid_ask(old_ask, old_bid, init_state, end_state, alpha[i])
        if number_of_actions > 0:
            if (profit - cost_factor_for1mwtrade * number_of_actions) / number_of_actions > best_fraction:
                best_fraction = profit / number_of_actions
                info_of_best = [new_allocation, profit, number_of_actions]
            new_profit_list.append(profit)
            new_alloc_list.append(new_allocation)
            num_actions_list.append(number_of_actions)
            fractions_list.append(profit / number_of_actions)

    number_of_action_cost = 10  # for 3 mw. 6eur normally for 1 mw buysell pair assuming 2.5million investment and 10mw battery
    results = pd.DataFrame(list(zip(new_alloc_list, new_profit_list, num_actions_list, fractions_list)), columns=['new_allocation', 'profit', 'number_of_actions', 'fraction'])
    results = results.drop_duplicates(subset=['profit'], keep='first')
    results['profit_diff'] = results.profit.diff()
    if results.profit.iloc[0] > number_of_action_cost:
        best_result = results[results.profit_diff > number_of_action_cost]
        best_result = best_result.iloc[[-1]]
    else:
        best_result = results.iloc[[0]]
        print('optimal_allocation profit worse than unit cost')

    end = time.time()
    duration = end - start
    initial_allocation, initial_profit, number_of_actions = best_result.new_allocation.iloc[0], best_result.profit.iloc[0], best_result.number_of_actions.iloc[0]

    return initial_allocation, np.round(initial_profit, 2), number_of_actions

"""
This function is discarded from get_initial_allocation2_threaded.
It is called no more because it simply calls allocate_bid_ask.
The allocate_bid_ask function call is transferred to get_initial_allocation2_threaded.
"""
def get_allocation(l,old_ask, old_bid, init_state, end_state, num, q):
    new_allocation, profit, number_of_actions = allocate_bid_ask(old_ask, old_bid, init_state, end_state, num)
    q.put([new_allocation, profit, number_of_actions])
    return new_allocation, profit, number_of_actions

"""
Before refactoring, this function was not multiprocessing ready. Instead it was calling
get_allocation, and virtually made get_allocation multi-processed. In the new version get_allocation
is eliminated and get_initial_allocation2_threaded is optimized to be multiprocessed.

It is a simple function that calls allocate_bid_ask, which does the heavy work, and stores the results
in a multiprocessing queue.
"""
def get_initial_allocation2_threaded(ask_dict, bid_dict, current_alpha, q, init_state=4, end_state=4, alpha=1,):

    old_ask = ask_dict
    old_bid = bid_dict

    new_allocation, profit, number_of_actions = allocate_bid_ask(old_ask, old_bid, init_state, end_state, current_alpha)
    q.put([new_allocation, profit, number_of_actions], block=False, timeout=5)
    return new_allocation, profit, number_of_actions

def get_new_allocation2(ask, bid, old_ask, old_bid, old_allocation, init_state, end_state, alpha=1):
    old_ask, old_bid, old_allocation = trim_old_allocation(old_ask, old_bid, ask, old_allocation)  # remove the keys that have expired

    n_alpha = 1000  # choose the appropriate number of alpha values
    alpha = np.linspace(0, 1.0, num=n_alpha)
    new_alloc_list, new_profit_list, num_actions_list, fractions_list = [], [], [], []
    best_fraction = -0.0001
    cost_factor_for1mwtrade = 0  # 6eur normally for 1 mw buysell pair assuming 2.5million investment and 10mw battery
    for i in range(0, n_alpha):
        new_allocation, profit, number_of_actions = allocate_bid_ask_new(ask, bid, old_ask, old_bid, old_allocation, init_state, end_state, alpha[i])
        if number_of_actions > 0:
            if (profit - cost_factor_for1mwtrade*number_of_actions)/number_of_actions > best_fraction :
                best_fraction = profit/number_of_actions
                info_of_best = [new_allocation, profit, number_of_actions]
            new_profit_list.append(profit)
            new_alloc_list.append(new_allocation)
            num_actions_list.append(number_of_actions)
            fractions_list.append(profit/number_of_actions)

    number_of_action_cost = 10  # for 3 mw. 6eur normally for 1 mw buysell pair assuming 2.5million investment and 10mw battery
    results = pd.DataFrame(list(zip(new_alloc_list, new_profit_list, num_actions_list, fractions_list)), columns=['new_allocation', 'profit', 'number_of_actions', 'fraction'])
    results = results.drop_duplicates(subset=['number_of_actions'], keep='last')
    results['profit_diff'] = results.profit.diff()

    if results.profit.iloc[0] > number_of_action_cost:
        best_result = results[results.profit_diff > number_of_action_cost]
        best_result = best_result.iloc[[-1]]
    else:
        best_result = results.iloc[[0]]
        print('optimal_allocation profit worse than unit cost')

    new_allocation, profit, number_of_actions = best_result.new_allocation.iloc[0], best_result.profit.iloc[0], best_result.number_of_actions.iloc[0]
    # profit  = allocation_profit + cost_profit
    allocation_profit, cost_profit = allocation_part_cost_part(old_ask, old_bid, ask, bid, old_allocation, new_allocation)
    next_old_ask, next_old_bid = get_true_current_bid_ask(old_ask, old_bid, ask, bid, old_allocation, new_allocation)

    return [[new_allocation, profit, number_of_actions]], [allocation_profit, cost_profit], [next_old_ask, next_old_bid, old_allocation]


def main():
    # create random bids and asks
    init_state = 4
    end_state = 4
    start_time = time.time()

    ##old_bid = {141: 5, 142: 13, 143: 6, 144: 3}

    old_ask = {141: 10, 142: 15, 143: 7, 144: 5, 151: 6, 152: 16}
    old_bid = {141: 5, 142: 13, 143: 6, 144: 3, 151: 1, 152: 13}

    #old_ask = {141: 10, 142: 15, 143: 7, 144: 5}
    #old_bid = {141: 5, 142: 13, 143: 6, 144: 3}

    #old_bid, old_ask, new_bid, new_ask = get_random_ask_bid(length=20)

    old_alloc, old_profit, _ = allocate_bid_ask(old_ask, old_bid, init_state, end_state, 1)

    # new dictionaries are created in a way that new keys are added in the end, while the bids and asks for old keys are updated
    #new_ask = {141: 15, 142: -30, 143: 11, 144: 1}
    #new_bid = {141: 14, 142: -40, 143: 10, 144: 0}

    new_ask = {141: 1, 142: 25, 143: 9, 144: 5, 151: 12, 152: 10}
    new_bid = {141: 5, 142: 3, 143: 6, 144: 9, 151: 3, 152: 4}


    old_ask, old_bid, old_alloc = trim_old_allocation(old_ask, old_bid, new_ask, old_alloc)
    new_alloc, new_profit, num_of_actions = allocate_bid_ask_new(new_ask, new_bid, old_ask, old_bid, old_alloc, init_state, end_state, alpha=1)

    # n_alpha = 100  # choose the appropriate number of alpha values
    # alpha = np.linspace(0, 1.0, num=n_alpha)
    # new_alloc_list, new_profit_list, num_actions_list = [], [], []
    # for i in range(0, n_alpha):
    #     new_alloc, new_profit, num_actions = allocate_bid_ask_new(new_ask, new_bid, old_ask, old_bid, old_alloc, init_state, end_state, alpha[i])
    #
    #     print("Iter:", i, "Max profit:", round(new_profit, 2), "Number of actions: ", num_actions)
    #     new_alloc_list.append(new_alloc)
    #     new_profit_list.append(new_profit)
    #     num_actions_list.append(num_actions)
    #
    # print("---------------------------------------")
    # print("Total duration:", time.time() - start_time)
    #
    # # plot maximum profit and number of actions
    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle('Maximum profit and number of actions for multiple values of alpha')
    # axs[0].plot(alpha, new_profit_list)
    # axs[0].set_xlabel('alpha')
    # axs[0].set_ylabel('maximum profit')
    # axs[0].grid()
    # axs[1].plot(alpha, num_actions_list)
    # axs[1].set_xlabel('alpha')
    # axs[1].set_ylabel('num actions')
    # axs[1].grid()
    # plt.show()
    # old_ask, old_bid, old_alloc = trim_old_allocation(old_ask, old_bid, new_ask, old_alloc)  # remove ask and bid keys which are not valid anymore


    print("Old allocation: ", old_alloc, "| Old profit: ", old_profit)
    print("New allocation: ", new_alloc, "| New profit: ", new_profit)  # print(len([1 for i in new_alloc if i == 1]))


def get_true_current_bid_ask(old_ask, old_bid, ask, bid, old_allocation, new_allocation):
    i = 0
    for key, value in ask.items():
        if new_allocation[i] * old_allocation[i] == 1 and new_allocation[i] == 1:
            ask[key] = old_ask[key]
        elif new_allocation[i] * old_allocation[i] == 1 and new_allocation[i] == -1:
            bid[key] = old_bid[key]
        i = i+1

    return ask, bid


def allocation_part_cost_part(old_ask_array, old_bid_array, ask_array, bid_array, old_allocation, new_allocation):
    x_opt = new_allocation
    old_alloc = old_allocation
    ask_array, bid_array = np.fromiter(ask_array.values(), dtype=float), np.fromiter(bid_array.values(), dtype=float)
    old_ask_array, old_bid_array = np.fromiter(old_ask_array.values(), dtype=float), np.fromiter(old_bid_array.values(), dtype=float)
    allocation_part = []
    cost_part = []
    # for open positions from old allocation: if cost is negative its costs, if its positive its profit
    for i in range(len(x_opt)):
        if x_opt[i] != 0:
            if x_opt[i] == 1:
                allocation_part.append(-ask_array[i])
            else:
                allocation_part.append(bid_array[i])
        else:
            allocation_part.append(0)

    for i in range(len(x_opt)):
        # if position is open before and stays the same, append the difference part, because essentially we want to evaluate the trading cost of the part cost.
        if old_alloc[i] == 1:
            if x_opt[i] == 1:
                allocation_part.append(ask_array[i] - old_ask_array[i])
            elif x_opt[i] == -1 or x_opt[i] == 0:
                cost_part.append(bid_array[i] - old_ask_array[i])
        elif old_alloc[i] == -1:
            if x_opt[i] == -1:
                allocation_part.append(-bid_array[i] + old_bid_array[i])
            elif x_opt[i] == 0 or x_opt[i] == 1:
                cost_part.append(old_bid_array[i] - ask_array[i])
        elif old_alloc[i] == 0:
            cost_part.append(0)

    # print(f'allocation_part: {np.round(sum(allocation_part), 2)} + cost/profit part: {np.round(sum(cost_part), 2)} == {np.round((sum(allocation_part) + sum(cost_part)),2)}')

    return np.round(sum(allocation_part), 2), np.round(sum(cost_part), 2)
