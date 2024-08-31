import cvxpy as cp
import numpy as np
import time
from matplotlib import pyplot as plt


# original allocation function, only for testing purposes


def allocate_bid_ask(ask_dict, bid_dict, init_cap, end_cap, min_buy_sell_cap, total_cap, cost):

    n = len(ask_dict)
    ask_array = np.fromiter(ask_dict.values(), dtype=float)
    bid_array = np.fromiter(bid_dict.values(), dtype=float)

    # encode capacities into numbers, since we work with integers
    init_count = int(init_cap/min_buy_sell_cap)
    end_count = int(end_cap/min_buy_sell_cap)
    total_count = int(total_cap/min_buy_sell_cap)

    # Construct a CVXPY problem
    M = total_count  # minimal "large" constant for bigM notation. Look at https://yalmip.github.io/tutorial/logicprogramming
    x_ask = cp.Variable(n, integer=True)
    x_bid = cp.Variable(n, integer=True)
    ask_flag = cp.Variable(n, boolean=True) # 0 if x_ask = 0, 1 if x_ask > 0
    bid_flag = cp.Variable(n, boolean=True) # 0 if x_bid = 0, 1 if x_bid > 0
    objective = cp.Maximize(min_buy_sell_cap * (bid_array @ x_bid - ask_array @ x_ask - cost*cp.sum((x_bid + x_ask)))) # spend/buy on the ask, earn/sell on the bid + fixed cost for each sell/bid
    constraints = [x_ask >= 0, x_ask <= total_count, x_bid >= 0, x_bid <= total_count,  # variables have to be constrained since they are now integers
                   cp.cumsum(x_ask - x_bid)[n-1] == end_count - init_count,  # init_count + cumulative sum of the actions has to equal end_cound
                   cp.cumsum(x_ask - x_bid) >= 0 - init_count,  # initial state + cumulative sum of the actions should always be larger than 0
                   cp.cumsum(x_ask - x_bid) <= total_count - init_count,  # initial state + cumulative sum of the actions should always be smaller than total_count
                   x_ask <= M*ask_flag,  # when x_ask is non-zero, ask_flag has to be 1 in order to satisfy the constraints
                   x_bid <= M*bid_flag,  # when x_abid is non-zero, bid_flag has to be 1 in order to satisfy the constraints
                   ask_flag + bid_flag <= 1]  # we are not allowed to buy and sell at the same time
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # extract the values
    max_profit = min_buy_sell_cap*(bid_array @ x_bid.value - ask_array @ x_ask.value)
    x_opt = min_buy_sell_cap * (x_ask.value - x_bid.value)
    num_actions = min_buy_sell_cap * np.sum(x_ask.value, dtype=int)

    # min_buy_sell_cap * x_ask.value, min_buy_sell_cap * x_bid.value

    return x_opt, max_profit, num_actions


# from old allocation remove the keys which are "expired"
def trim_old_allocation(old_ask, old_bid, new_ask, old_alloc):

    n_old = len(old_ask) # number of old bids and asks
    # we assume the time steps are correctly ordered and that bid and ask dictionaries have the same keys!!!
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


def allocate_bid_ask_new(ask, bid, old_ask, old_bid, old_alloc, init_cap, end_cap, min_buy_sell_cap, total_cap, cost):

    n, n_old = len(ask), len(old_ask)
    ask_array, bid_array = np.fromiter(ask.values(), dtype=float), np.fromiter(bid.values(), dtype=float)
    old_ask_array, old_bid_array = np.fromiter(old_ask.values(), dtype=float), np.fromiter(old_bid.values(), dtype=float)

    old_bid_flag =  np.array([int(item) for item in old_alloc < 0])  # extract old bid allocation indicator (logical)
    old_ask_flag =  np.array([int(item) for item in old_alloc > 0])  # extract old ask allocation indicator (logical)
    old_x_bid = 1/min_buy_sell_cap * np.multiply(np.abs(old_alloc), old_bid_flag)  # extract old bid allocation (true value)
    old_x_ask = 1/min_buy_sell_cap * np.multiply(np.abs(old_alloc), old_ask_flag)  # extract old ask allocation (true value)

    # encode capacities into numbers, since we work with integers
    init_count = int(init_cap/min_buy_sell_cap)
    end_count = int(end_cap/min_buy_sell_cap)
    total_count = int(total_cap/min_buy_sell_cap)

    # Construct a CVXPY problem
    M = total_count  # minimal "large" constant for bigM notation. Look at, for example https://yalmip.github.io/tutorial/logicprogramming
    x_ask = cp.Variable(n, integer=True)
    x_bid = cp.Variable(n, integer=True)
    ask_flag = cp.Variable(n, boolean=True)  # 0 if x_ask = 0, 1 if x_ask > 0
    bid_flag = cp.Variable(n, boolean=True)  # 0 if x_bid = 0, 1 if x_bid > 0

    # CURRENT FLAGS AND PREVIOUS VALUES
    bought_before_value_sold_now_flag = cp.multiply(bid_flag[0:n_old], old_x_ask)
    sold_before_value_bought_now_flag = cp.multiply(ask_flag[0:n_old], old_x_bid)
    bought_before_value_na_now = cp.multiply(1 - bid_flag[0:n_old] - ask_flag[0:n_old], old_x_ask) 
    sold_before_value_na_now = cp.multiply(1 - bid_flag[0:n_old] - ask_flag[0:n_old], old_x_bid)
    bought_before_value_bought_now_flag = cp.multiply(ask_flag[0:n_old], old_x_ask) 
    sold_before_value_sold_now_flag = cp.multiply(bid_flag[0:n_old], old_x_bid)

    # bought_before_bought_more_now_flag = cp.multiply(ask_flag[0:n_old] - old_x_ask, ask_flag[0:n_old])
    # sold_before_sold_more_now_flag =
    bought_before_bought_more_now = ask_array[0:n_old] @ bought_before_value_bought_now_flag
    sold_before_sold_more_now = - bid_array[0:n_old] @ sold_before_value_sold_now_flag

    # PREVIOUS FLAGS AND CURRENT VALUES
    bought_before_flag_sold_now_value = cp.multiply(x_bid[0:n_old], old_ask_flag)
    sold_before_flag_bought_now_value = cp.multiply(x_ask[0:n_old], old_bid_flag)
    bought_before_flag_bought_now_value = cp.multiply(x_ask[0:n_old], old_ask_flag)
    sold_before_flag_sold_now_value = cp.multiply(x_bid[0:n_old], old_bid_flag)

    objective = cp.Maximize( min_buy_sell_cap * ((bid_array @ x_bid - ask_array @ x_ask) - cost*cp.sum((x_bid + x_ask)) +  #nominal profit
                                                (old_bid_array - ask_array[0:n_old]) @ sold_before_value_na_now +  # -1 -> 0
                                                (bid_array[0:n_old] - old_ask_array) @ bought_before_value_na_now +  # 1 -> 0
                                                old_bid_array @ sold_before_value_bought_now_flag - ask_array[0:n_old] @ sold_before_flag_bought_now_value +  # -1 -> 1
                                                bid_array[0:n_old] @ bought_before_flag_sold_now_value - old_ask_array @ bought_before_value_sold_now_flag +  # 1 -> -1
                                                ask_array[0:n_old] @ bought_before_flag_bought_now_value - old_ask_array @ bought_before_value_bought_now_flag +  # 1 -> 1
                                                old_bid_array @ sold_before_value_sold_now_flag - bid_array[0:n_old] @ sold_before_flag_sold_now_value))  # -1 -> -1

    #  the 4 component corresponds to the 15min market interval. any move affects the inventory by move/4
    constraints = [x_ask >= 0, x_ask <= total_count, x_bid >= 0, x_bid <= total_count,  # variables have to be constrained since they are now integers
                   cp.cumsum(x_ask - x_bid)[n-1] == 4*(end_count - init_count),  # init_count + cumulative sum of the actions has to equal end_count
                   cp.cumsum(x_ask - x_bid) >= 0 - 4*init_count,  # initial state + cumulative sum of the actions should always be larger than 0
                   cp.cumsum(x_ask - x_bid) <= 4*(total_count - init_count),  # initial state + cumulative sum of the actions should always be smaller than total_count
                   x_ask <= M*ask_flag,  # when x_ask is non-zero, ask_flag has to be 1 in order to satisfy the constraints
                   x_bid <= M*bid_flag,  # when x_abid is non-zero, bid_flag has to be 1 in order to satisfy the constraints
                   ask_flag + bid_flag <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # extract the values
    max_profit = objective.value
    x_opt = min_buy_sell_cap * (x_ask.value - x_bid.value)  # the allocation result is on units of 1mw
    num_actions = min_buy_sell_cap * (x_ask.value - x_bid.value)

    return x_opt, max_profit, num_actions


def main():

    init_cap = 1
    end_cap = 1
    min_buy_sell_cap = 1
    total_cap = 1
    cost = 0

    # create random bids and asks
    old_ask = {141: 10, 142: 15, 143: 7, 144: 5, 151: 6, 152: 16}
    old_bid = {141: 5, 142: 13, 143: 6, 144: 3, 151: 1, 152: 13}

    #old_ask = {141: 10, 142: 15, 143: 7, 144: 5}
    #old_bid = {141: 5, 142: 13, 143: 6, 144: 3}

    old_alloc, old_profit, num_of_actions= allocate_bid_ask(old_ask, old_bid, init_cap, end_cap, min_buy_sell_cap, total_cap, cost)

    # new dictionaries are created in a way that new keys are added in the end, while the bids and asks for old keys are updated
    new_ask = {141: 1, 142: 25, 143: 9, 144: 5, 151: 12, 152: 10}
    new_bid = {141: 5, 142: 3, 143: 6, 144: 9, 151: 3, 152: 4}

    #new_ask = {141: 15, 142: -30, 143: 11, 144: 1}
    #new_bid = {141: 14, 142: -40, 143: 10, 144: 0}

    old_ask, old_bid, old_alloc = trim_old_allocation(old_ask, old_bid, new_ask, old_alloc) # remove ask and bid keys which are not valid anymore
    new_alloc, new_profit, num_of_actions = allocate_bid_ask_new(new_ask, new_bid, old_ask, old_bid, old_alloc, init_cap, end_cap, min_buy_sell_cap, total_cap, cost)

    print("Old allocation: ", old_alloc, "| Old profit: ", old_profit)
    print("New allocation: ", new_alloc, "| New profit: ", new_profit)

    x = allocation_part_cost_part(old_ask, old_bid, new_ask, new_bid, old_alloc, new_alloc)
    y=2




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


def get_initial_allocation(ask_dict, bid_dict, init_cap, end_cap, min_buy_sell_cap, total_cap, cost):
    old_ask = ask_dict
    old_bid = bid_dict

    initial_allocation, initial_profit, number_of_actions = allocate_bid_ask(old_ask, old_bid, init_cap, end_cap, min_buy_sell_cap, total_cap, cost)

    return initial_allocation, np.round(initial_profit, 2), number_of_actions


def get_new_allocation(ask, bid, old_ask, old_bid, old_allocation, init_cap, end_cap, min_buy_sell_cap, total_cap, cost):
    old_ask, old_bid, old_allocation = trim_old_allocation(old_ask, old_bid, ask, old_allocation)  # remove the keys that have expired
    new_allocation, profit, number_of_actions = allocate_bid_ask_new(ask, bid, old_ask, old_bid, old_allocation, init_cap, end_cap, min_buy_sell_cap, total_cap, cost)

    # profit  = allocation_profit + cost_profit
    allocation_profit, cost_profit = allocation_part_cost_part(old_ask, old_bid, ask, bid, old_allocation, new_allocation)
    next_old_ask, next_old_bid = get_true_current_bid_ask(old_ask, old_bid, ask, bid, old_allocation, new_allocation)

    return [[new_allocation, profit, number_of_actions]], [allocation_profit, cost_profit], [next_old_ask, next_old_bid, old_allocation]


def get_true_current_bid_ask(old_ask, old_bid, ask, bid, old_allocation, new_allocation):
    i = 0
    for key, value in ask.items():
        if new_allocation[i] > 0:
            if new_allocation[i] * old_allocation[i] > 0 and new_allocation[i] <= old_allocation[i]:
                ask[key] = old_ask[key]
            elif new_allocation[i] * old_allocation[i] > 0 and new_allocation[i] > old_allocation[i]:
                p = old_allocation[i]/new_allocation[i]
                ask[key] = old_ask[key] * p + ask[key] * (1-p)
        elif new_allocation[i] < 0:
            if new_allocation[i] * old_allocation[i] > 0 and new_allocation[i] >= old_allocation[i]:
                bid[key] = old_bid[key]
            elif new_allocation[i] * old_allocation[i] > 0 and new_allocation[i] < old_allocation[i]:
                p = old_allocation[i] / new_allocation[i]
                bid[key] = old_bid[key] * p + bid[key] * (1-p)
        i = i + 1

    return bid, ask


def allocation_part_cost_part(old_ask_array, old_bid_array, ask_array, bid_array, old_allocation, new_allocation):
    x_opt = new_allocation
    old_alloc = old_allocation
    ask_array, bid_array = np.fromiter(ask_array.values(), dtype=float), np.fromiter(bid_array.values(), dtype=float)
    old_ask_array, old_bid_array = np.fromiter(old_ask_array.values(), dtype=float), np.fromiter(old_bid_array.values(), dtype=float)
    allocation_part = []
    cost_part = []
    """
    the idea for the cases where we increase positions is as following, example for the long case:
    allocation_Part[i] = - new_ask[i] * new_allocation[i] (normal)
    cost_part[i] = old_allocation[i] * old_ask[i] + old_allocation[i]*[new_ask[i] - old_ask[i]] = old_allocation[i] * new_ask [i]
    if we sum the allocation and the cost we get:
    total[i] = - new_ask[i] * ( new_allocation[i] - old_allocation[i]), which is what we paid in reality to increase the position
    """

    for i in range(len(x_opt)):
        if x_opt[i] != 0:
            if new_allocation[i] > 0:
                allocation_part.append(-ask_array[i] * np.abs(x_opt[i]))
            else:
                allocation_part.append(bid_array[i] * np.abs(x_opt[i]))
        else:
            allocation_part.append(0)

    for i in range(len(new_allocation)):

        if old_allocation[i] > 0:
            if new_allocation[i] >= 0:
                mw_diff = np.abs(new_allocation[i] - old_allocation[i])
                if new_allocation[i] < old_allocation[i]:
                    cost_part.append((bid_array[i] - old_ask_array[i]) * mw_diff)
                elif new_allocation[i] > old_allocation[i]:
                    # here is the part where we increase the position
                    cost_part.append(old_allocation[i]*ask_array[i])
                elif new_allocation[i] == old_allocation[i]:
                    allocation_part.append((ask_array[i] - old_ask_array[i]) * new_allocation[i])
            else:
                mw_diff = np.abs(old_allocation[i])
                cost_part.append((bid_array[i] - old_ask_array[i]) * mw_diff)
        elif old_allocation[i] < 0:
            if new_allocation[i] <= 0:
                mw_diff = np.abs(new_allocation[i] - old_allocation[i])
                if new_allocation[i] > old_allocation[i]:
                    cost_part.append((-ask_array[i] + old_bid_array[i]) * mw_diff)
                elif new_allocation[i] < old_allocation[i]:
                    # here is the part where we increase the position, being short
                    cost_part.append(-old_allocation[i]*bid_array[i])
                elif new_allocation[i] == old_allocation[i]:
                    allocation_part.append((-bid_array[i] + old_bid_array[i]) * np.abs(new_allocation[i]))
            else:
                mw_diff = np.abs(old_allocation[i])
                cost_part.append((-ask_array[i] + old_bid_array[i]) * mw_diff)

        # print(f'allocation_part: {np.round(sum(allocation_part), 2)} + cost/profit part: {np.round(sum(cost_part), 2)} == {np.round((sum(allocation_part) + sum(cost_part)),2)}')

    return np.round(sum(allocation_part), 2), np.round(sum(cost_part), 2)


if __name__ == "__main__":
    main()