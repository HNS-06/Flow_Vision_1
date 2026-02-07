
import pulp
import pandas as pd
import numpy as np

class OptimizationEngine:
    """
    Optimizes water distribution using Linear Programming.
    Objective: Minimize (Pumping Cost + Shortage Penalty)
    """

    def __init__(self):
        pass

    def optimize_distribution(self, ward_data, total_supply):
        """
        Calculate optimal flow allocation to wards.
        
        ward_data: List of dicts containing:
            - ward_id
            - demand (L/min)
            - current_level (0-100%)
            - capacity (L)
            - priority (1-5, 5 is critical)
            
        total_supply: Total water available (L/min)
        """
        
        # Create LP Problem
        prob = pulp.LpProblem("Water_Redistribution", pulp.LpMinimize)
        
        # Variables: Flow to each ward (>= 0)
        ward_ids = [w['ward_id'] for w in ward_data]
        flows = pulp.LpVariable.dicts("Flow", ward_ids, lowBound=0)
        
        # Shortage variables (>= 0) - Amount of demand NOT met
        shortages = pulp.LpVariable.dicts("Shortage", ward_ids, lowBound=0)
        
        # Parameters
        cost_per_liter = 0.05
        penalty_per_liter_shortage = 10.0 # High penalty for shortage
        
        # Objective Function: Minimize Cost + Shortage Penalties (weighted by priority)
        prob += pulp.lpSum([
            (flows[w['ward_id']] * cost_per_liter) + 
            (shortages[w['ward_id']] * penalty_per_liter_shortage * w.get('priority', 1)) 
            for w in ward_data
        ])
        
        # Constraints
        
        # 1. Supply Limit: Total flow out <= Total Supply
        prob += pulp.lpSum([flows[w_id] for w_id in ward_ids]) <= total_supply, "TotalSupplyConstraint"
        
        # 2. Demand Balance: Flow + Shortage = Demand (Ensure demand is accounted for)
        for w in ward_data:
            w_id = w['ward_id']
            prob += flows[w_id] + shortages[w_id] == w['demand'], f"DemandConstraint_{w_id}"
            
        # 3. Capacity Constraints (Logical check - can't overfill)
        # We assume flow is instant for this simplified model, but in reality depends on time step.
        # Here we just ensure we don't assign flow > max intake rate (e.g., pipe capacity 500 L/min)
        max_pipe_capacity = 500
        for w_id in ward_ids:
            prob += flows[w_id] <= max_pipe_capacity, f"PipeCap_{w_id}"
            
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract Results
        results = []
        status = pulp.LpStatus[prob.status]
        
        for w in ward_data:
            w_id = w['ward_id']
            allocated = pulp.value(flows[w_id])
            shortage = pulp.value(shortages[w_id])
            demand = w['demand']
            
            results.append({
                'ward_id': w_id,
                'demand': demand,
                'allocated_flow': round(allocated, 2),
                'shortage': round(shortage, 2),
                'status': 'Optimal' if shortage < 0.1 else 'Constrained'
            })
            
        return {
            'status': status,
            'total_supply': total_supply,
            'allocations': results,
            'optimization_score': round(pulp.value(prob.objective), 2)
        }

if __name__ == "__main__":
    # Test
    engine = OptimizationEngine()
    wards = [
        {'ward_id': 1, 'demand': 150, 'priority': 3},
        {'ward_id': 2, 'demand': 200, 'priority': 5}, # Industrial - Critical
        {'ward_id': 3, 'demand': 80, 'priority': 1},
    ]
    supply = 350 # Shortage scenario (Demand is 430)
    
    print(engine.optimize_distribution(wards, supply))
