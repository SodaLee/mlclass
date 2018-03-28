nums = eval(input())
k = int(input())
s = set(nums)
d = dict([[a, nums.count(a)] for a in s])
s1 = sorted(d.items(), key=lambda d: d[1], reverse=True)
s2 = [e[0] for e in s1]
print(s2[0:k])