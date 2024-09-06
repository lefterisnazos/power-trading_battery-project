# Power Trading Battery based Project

A project focused on optimizing battery allocation across different auction and continuous markets. This includes the continuous trading of the battery in the market using a solution based on convex optimization. Due to various live data, package, and database dependencies, the files are not currently runnable. Core related files are located in the 'battery_allocator' file

## Project Structure

### Continuous Market

- `allocation_optimizer.py` & `allocation_optimizer_parametrized.py`
  Responsible for optimal allocation of the battery in the continuous 15-minute market brackets, given current environment conditions such as battery level, prices, etc.

- `xbid_allocation_live.py`  
  The interface that facilitates trading of the battery in the continuous market.

- `xbid_automated_runner.py` & `xbid_runner.py`
  These scripts are responsible for calling a parametrized instance of the main function in "xbid_allocation_live.py", enabling automated trading of the battery in the continuous market for a specified period.

### Auction Markets

- `auction_15_production.py` & `fcr_vs_da_production.py`
  Responsible for the daily allocation of the battery in the auction markets.

- `production_auction_markets.py`
  Executes a daily run to produce the schedule for the auction markets.

### other

- `battery_simulator.py` 
  An initial attempt to create a smart object representing the properties of the battery, intended for use in eventual trading scenarios.
