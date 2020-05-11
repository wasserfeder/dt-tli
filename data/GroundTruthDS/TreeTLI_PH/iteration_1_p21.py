
from ana_STL import STL_computation
from ana_STL import directed_distance

F=STL_computation(2,100)

f_0 = F.add_predicate(1,"<",0.1)
f_1 = F.add_predicate(2,"<",0.6)
f_2 = F.Conj([f_0, f_1])
f_3 = F.add_predicate(2,">=",0.4)
f_4 = F.Conj([f_2, f_3])
f_5 = F.G(range(0, 10+1), f_4)
f_6 = F.add_predicate(1,">=",0.7)
f_7 = F.add_predicate(2,">=",0.8)
f_8 = F.add_predicate(2,"<",0.2)
f_9 = F.Disj([f_7, f_8])
f_10 = F.Conj([f_6, f_9])
f_11 = F.G(range(70, 100+1), f_10)
f_12 = F.Conj([f_5, f_11])
f_13 = F.add_predicate(1,"<=",0.65)
f_14 = F.F(range(0, 41+1), f_13)
f_15 = F.G(range(32, 52+1), f_14)
f_16 = F.add_predicate(2,"<=",0.64)
f_17 = F.F(range(0, 4+1), f_16)
f_18 = F.G(range(6, 8+1), f_17)
f_19 = F.Conj([f_15, f_18])

r=directed_distance(F, f_19, f_12)
print(r)

