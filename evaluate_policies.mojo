from testing import assert_true
from mojo_phi.food_truck.food_truck import FoodTruck, base_policy, policy_evaluation, State

def print_list(x: List[State]):
    for e in x:
        elm = e[]
        print(elm)

def evaluate_policies():
    foodtruck = FoodTruck()
    policy = base_policy(foodtruck.state_space)

    assert_true(State(0, 0) in policy)
    value = policy_evaluation(foodtruck, policy, max_iter=100)
    for kv in value.items():
        key = kv[].key
    week_profit = value[State(0, 0)]
    print("Expected weekly profit: ", str(week_profit))

def main():
    evaluate_policies()
