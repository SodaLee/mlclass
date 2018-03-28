nums = eval(input())
def ans(lis):
	if (len(lis)>0):
		ret = ans(lis[1:]) + ans(lis[1:])
		for e in ret[len(ret)//2:]:
			e.append(lis[0])
		return ret
	else:
		return list([[]])
print(ans(nums))