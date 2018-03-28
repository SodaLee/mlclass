import weibo

if __name__ == "__main__":
	for i in range(10):
		file = open("ans.txt", "a")
		file.write("round %d\n" % i)
		ans = weibo.train()
		file.write("correctness %f\n" % ans)