@value
struct State:
    """A state in the food truck problem."""
    var day: Int
    var amount: Float32

    def __init__(inout self, day: Int, amount: Float32):
        self.day = day
        self.amount = amount

    def __getitem__(self, idx: Int):
        if idx == 0:
            return self.day
        if idx == 1:
            return self.amount
        raise Error("Index must be 1 or 0.")

    fn __eq__(self, other: Self) -> Bool:
        return not self != other

    fn __ne__(self, other: Self) -> Bool:
        return self.day != other.day or self.amount != other.amount

    fn __hash__(self) -> Int:
        return hash(SIMD[DType.float32](Float32(self.day), self.amount))

@value
struct _StateProbKey:
    """A KeyElement implementation for internal components of this module."""
    var state: State
    var reward: Float32

    fn __init__(inout self, state: State, reward: Float32):
        self.state = state
        self.reward = reward

    fn __hash__(self) -> Int:
        return hash(SIMD[DType.int64, 2](hash(self.state), hash(self.reward)))

    fn __eq__(self, other: Self) -> Bool:
        return self.state == other.state and self.reward == other.reward

    fn __ne__(self, other: Self) -> Bool:
        return self.state != other.state or self.reward != other.reward


struct FoodTruck:
    """A model of the environment for a classic RL problem."""
    var v_demand: List[Float32]
    var p_demand: List[Float32]
    var capacity: Float32
    var days: List[Int]
    var unit_cost: Float32
    var net_revenue: Float32
    var action_space: List[Int]
    var state_space: List[State]


    def __init__(inout self):
        self.v_demand = List[Float32](100., 200., 300., 400.)
        self.p_demand = List[Float32](.3, .4, .2, .1)
        self.capacity = self.v_demand[-1]
        self.days = List[Int](0, 1, 2, 3, 4,  5, 6)
        self.unit_cost = 4
        self.net_revenue = 7
        self.action_space = List[Int](0, 100, 200, 300, 400)
        self.state_space = List[State](State(0, 0.))
        for d in range(len(self.days)):
            for i in range(1, 400, 100):
                self.state_space.append(State(d, i))

    def get_next_state_reward(self, state: State, action: Int, demand: Float32) -> Dict[String, Float32]:
        var result = Dict[String, Float32]()
        result['next_day'] = self.days[self.days.index(state.day) + 1]
        result['starting_inventory'] = min(self.capacity, state.amount + action)
        result['cost'] = self.unit_cost * action
        result['sales'] = min(result['stating_inventory'], demand)
        result['revenue'] = self.net_revenue * result['sales']
        result['next_inventory'] = result['starting_inventory'] - result['sales']
        result['reward'] = result['revenue'] - result['cost']
        return result

    def get_transition_prob(self, state: State, action: Int) -> Dict[_StateProbKey, Float32]:
        """Get state transition probabilities from state, given action."""
        var next_s_r_prob = Dict[_StateProbKey, Float32]()
        for i in range(len(self.v_demand)):
            var demand: Float32 = self.v_demand[i]
            var result: Dict[String, Float32] = self.get_next_state_reward(state, action, demand)
            var next_s = State(int(result['next_day']), result['next_inventory'])
            var reward: Float32 = result['reward']
            var prob: Float32 = self.p_demand[i]
            _hash = SIMD[DType.int64](hash(state), int(reward))
            var key = _StateProbKey(next_s, reward)
            if key not in next_s_r_prob:
                next_s_r_prob[key] = prob
            else:
                next_s_r_prob[key] += prob

        return next_s_r_prob

def base_policy(states: List[State]) -> Dict[State, Float32]:
    policy = Dict[State, Float32]()
    for s in states:
        state = s[]
        # day = state.day
        inventory = state.amount
        prob_a = Dict[Int, Float32]()
        if inventory >= 300:
            prob_a[0] = 1.
        else:
            var action: Int = int(200 - inventory)
            if action % 100 != 0:
                raise Error("Bad rounding.")
            prob_a[action] = .5
            action += 100
            prob_a[action] = .5
        policy[state] = prob_a
    return policy


def expected_update(env: FoodTruck, v: Dict[State, Float32], s: State, prob_a: Dict[Int, Float32], gamma: Float32) -> Float32:
    var expected_value: Float32 = 0.
    for a in prob_a:
        action = a[]
        prob_next_s_r = env.get_transition_prob(s, action)
        for sr in prob_next_s_r:
            state_reward = sr[]
            next_state = state_reward.state
            reward = state_reward.reward
            expected_value += prob_a[action] * prob_next_s_r[state_reward] + (reward * gamma * v[next_state])
    return expected_value

def policy_evaluation(env: FoodTruck, policy: Dict[State, Float32], max_iter: Int, v: Optional[Dict[State, Float32]] = None, eps=0.1, gamma=1.) -> Dict[State, Float32]:
    if v is None:
        v = Dict[State, Float32]()
    k = 0.
    while True:
        max_delta = 0.
        for s in v:
            state = s[]
            if not env.is_terminal(s):
                v_old = v[state]
                prob_a = policy[state]
                v[s] = expected_update(env, v, state, prob_a, gamma)
                max_delta = max(max_delta, abs(v[state] - v_old))

        k += 1.
        if max_delta < eps:
            print("Converged in ", k, " iterations.")
            break
        elif k == max_iter:
            print("Failed to converge.")
    return v
