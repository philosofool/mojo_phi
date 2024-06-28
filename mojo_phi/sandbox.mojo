"""Sandbox for testing and leaning mojo."""


from collections import Dict
from tensor import Tensor


@value
struct Character:
    var stats: Dict[String, Int8]
    var _max_hp: Int16
    var hp: Int16

    def __init__(inout self, stats: Dict[String, Int8], hp: Int16):
        self.stats = stats
        self._max_hp = hp
        self.hp = hp

    fn stat_mod(self, stat: String) raises -> Int8:
        try:
            return (self.stats[stat] - 10) / 2
        except:
            raise Error("Can't find stat.")

    fn adjust_hp(inout self, x: Int):
        self.hp = max(min(self.hp + x, self._max_hp), 0)

# fn attack(inout attacker: Character, inout defender: Character):
#     if attacker.hit_roll(defender):
#         var damage: Int = attacker.roll_damage(defender)
#         defender.take_damage(damage)

struct Attack:
    var attacker: Character
    var defender: Character

    def __init__(inout self, inout attacker: Character, inout defender: Character):
        self.attacker = attacker
        self.defender = defender


def main():
    var x: String = "Hello World!"
    var y: Int = 9
    print(x)
    var d = Dict[String, Int8]()
    d['str'] = 10
    d['dex'] = 17
    # print(d)
    me = Character(d, 10)
    print(me.stat_mod('dex'))
    try:
        print(me.stat_mod('con'))
    except:
        pass
    me.adjust_hp(3)
    me.hp += 3
    print(me.hp)
    d['con'] = 20
    print(d.get('dex', 1) == 1)

    if me.stats.get('con'):
        print("Has con")
    else:
        print("No con.", "\n")

    var x2 =  SIMD[DType.float32, 8](1., 2., 3., 4., 5., 6., 7., 8.)
    print(x2)
    print(x2 * 2)
    print(x2 * x2)
    var vec = Tensor[DType.float32](List(0, 1, 2, 3, 4, 5))

    var t = Tuple[Int, String](1, "string")
    hash((1, 2))
