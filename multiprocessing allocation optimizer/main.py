import math
import multiprocessing
import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Process
from allocation_optimizer2 import allocate_bid_ask
from allocation_optimizer2 import allocate_bid_ask_new
from allocation_optimizer2 import get_random_ask_bid
from allocation_optimizer2 import get_initial_allocation2_threaded
import cvxopt

if __name__ == '__main__':

    # pip3 install cvxopt
    # print(cvxpy.installed_solvers()) # execute this to ensure cvxopt is installed on system

    # start timing the operations
    start = time.time()

    """
    Necesary declarations for function calls.
    These declarations are partly taken from main function in allocation_optimizer.
    These can be wrong, so act accoringly. 
    """

    init_state = 4
    end_state = 4
    old_bid, old_ask, new_bid, new_ask = get_random_ask_bid()
    old_alloc, old_profit, _ = allocate_bid_ask(old_ask, old_bid, init_state, end_state, 1)
    new_alloc, new_profit, num_of_actions = allocate_bid_ask_new(new_ask, new_bid, old_ask, old_bid, old_alloc, init_state,end_state, alpha=1)

    """
    n_alpha and parallelism can be tweaked for a more aggressive or a more well-balanced resource consumption.
    parallelism: how many processes are allowed to run simultaneously. (%60 of total cores on the system by default)
    alpha_start/end: records for keeping track of the current and next alpha values    
    """

    n_alpha = 1000  # choose the appropriate number of alpha values
    alpha = np.linspace(0, 1.0, num=n_alpha)
    parallelism = math.floor(os.cpu_count() * .6)
    alpha_start= 0
    alpha_end = alpha_start + parallelism
    new_allocation, profit, number_of_actions = None, None, None
    new_alloc_list, new_profit_list, num_actions_list, fractions_list = [], [], [], []
    info_of_best = None
    best_fraction = -0.0001
    cost_factor_for1mwtrade = 0

    # initiate a multiprocessing queue
    q = multiprocessing.Queue(maxsize=n_alpha)

    # main process list, left here for debugging, has no active effect.
    processes = []
    count = 0

    # run until all n_alpha values are processed
    while count < n_alpha:

        # call get_initial_allocation2_threaded with current alpha value
        newprocesses = [Process(target=get_initial_allocation2_threaded, args=(old_ask, old_bid, alpha[i], q, init_state, end_state,)) for i in range(alpha_start, alpha_end)]
        processes += newprocesses

        # block if there is no room for new processes
        for process in newprocesses:
            if len(multiprocessing.active_children()) < parallelism:
                print("###### Starting process : " + str(count + 1))
                process.start()
                count = count + 1
            else:
                print("Waiting for processes to die...")
                time.sleep(1)

        """
        Wait for the current batch to finish.
        Check if the multiprocessing queue is no empty. If so, process the results in
        the queue until the queue is empyt.
        If the queue is empty break the loop, so that a new batch can be created.
        """
        while 1:
            running = any(p.is_alive() for p in newprocesses)
            while not q.empty():
                new_allocation, profit, number_of_actions = q.get()
                if number_of_actions > 0:
                     if (profit - cost_factor_for1mwtrade * number_of_actions) / number_of_actions > best_fraction:
                         best_fraction = profit / number_of_actions
                         info_of_best = [new_allocation, profit, number_of_actions]
                     new_profit_list.append(profit)
                     new_alloc_list.append(new_allocation)
                     num_actions_list.append(number_of_actions)
                     fractions_list.append(profit / number_of_actions)
            if not running:
                break

        # recalculate start and end records
        alpha_start = alpha_start + parallelism
        if (alpha_start + parallelism) > n_alpha:
            alpha_end = n_alpha
        else:
            alpha_end = alpha_start + parallelism

    # classic code flow from this point
    number_of_action_cost = -1  # for 3 mw. 6eur normally for 1 mw buysell pair assuming 2.5million investment and 10mw battery
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

    # Print Summary
    initial_allocation, initial_profit, number_of_actions = best_result.new_allocation.iloc[0], best_result.profit.iloc[0], best_result.number_of_actions.iloc[0]
    print("Total elapsed time: %.2f seconds." % duration)
    print("Result of the total execution:")
    print("Initial Allocation: " + str(initial_allocation))
    print("Initial Profit Result: " + str(np.round(initial_profit, 2)))
    print("Number of Actions: " + str(number_of_actions))
    print("Old allocation: ", old_alloc, "\nOld profit: ", old_profit)
    print("New allocation: ", new_alloc, "\nNew profit: ", new_profit)  # print(len([1 for i in new_alloc if i == 1]))