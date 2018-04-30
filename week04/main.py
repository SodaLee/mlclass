import cv2 #opencv_python
import numpy as np
drawing = False
MODE = True
v = [0, 0]
st, en = 0, 0
tot = 1

class Node(object):
	def __init__(self, s, e, c, nex):
		self.s = int(s)
		self.e = int(e)
		self.c = float(c)
		self.nex = int(nex)

def ADD(x, y, z1, z2):
	global tot
	global v
	global p
	global now
	tot += 1
	v.append(Node(x, y, z1, p[x]))
	p[x] = tot
	now[x] = tot
	tot += 1
	v.append(Node(y, x, z2, p[y]))
	p[y] = tot
	now[y] = tot

def SAP(k, water):
	global h
	global vh
	global now
	global v
	global p
	global en
	global eps
	if k == en:
		return water
	ret = 0
	q = now[k]
	while q:
		if h[v[q].e] + 1 == h[k] and v[q].c > 0:
			tmp = SAP(v[q].e, min(v[q].c, water - ret))
			ret += tmp
			v[q].c -= tmp
			v[q^1].c += tmp
			if ret == water:
				return ret
		q = v[q].nex
		now[k] = q
	vh[h[k]] -= 1
	if not vh[h[k]]:
		h[0] = en + 1
	h[k] += 1
	vh[h[k]] += 1
	now[k] = p[k]
	return ret

def BFS(queue):
	global fg_mask
	global p
	global v
	global st
	global en
	while len(queue) > 0:
		k = queue[0]
		del queue[0]
		q = p[k]
		while q:
			e = v[q].e
			if e != st and e != en:
				x = (e - 1) // fg_mask.shape[1]
				y = (e - 1) % fg_mask.shape[1]
				if v[q].c > 0 and fg_mask[x][y] == 0:
					fg_mask[x][y] = 1
					queue.append(e)
			q = v[q].nex

def simi(c1, c2):
	return float(c1[0]-c2[0])**2 + float(c1[1]-c2[1])**2 + float(c1[2]-c2[2])**2 + 0.1

def GraphCut(img, mask):
	global st
	global en
	global vh
	global h
	global p
	global now
	global fg_mask
	lamb1 = 20.0
	lamb2 = 2.0
	fg = np.zeros_like(img)
	bg = fg.copy()
	fg_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
	en = img.shape[0] * img.shape[1] + 1
	p = np.zeros(en + 5, np.int32)
	now = np.zeros(en + 5, np.int32)
	h = np.zeros(en + 5, np.int32)
	vh = np.zeros(en + 5, np.int32)
	fg_c = np.zeros(3)
	bg_c = np.zeros(3)
	fg_p, bg_p = 0, 0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if mask[i][j] == 0:
				fg_p += 1
				fg_c += img[i][j]
			elif mask[i][j] == 255:
				bg_p += 1
				bg_c += img[i][j]
	fg_c /= fg_p
	bg_c /= bg_p
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			num = i * img.shape[1] + j + 1
			if mask[i][j] == 0:
				ADD(st, num, 1e5, 0)
			elif mask[i][j] == 255:
				ADD(num, en, 1e5, 0)
			else:
				ADD(st, num, lamb1/simi(img[i][j], fg_c), 0)
				ADD(num, en, lamb1/simi(img[i][j], bg_c), 0)
			if i > 0:
				num1 = num - img.shape[1]
				c = lamb2/simi(img[i][j], img[i-1][j])
				ADD(num, num1, c, c)
			if j > 0:
				num1 = num - 1
				c = lamb2/simi(img[i][j], img[i][j-1])
				ADD(num, num1, c, c)
	print("Build Graph Finished")

	vh[0] = en + 1
	while h[0] <= en:
		print("SAP %f" % (SAP(int(st), 1e50)))
	print("SAP Finished")

	queue = []
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if mask[i][j] == 0:
				fg_mask[i][j] = 1
				queue.append(i * img.shape[1] + j + 1)
	BFS(queue)
	print("BFS Finished")
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if fg_mask[i][j]:
				for k in range(3):
					fg[i][j][k] = img[i][j][k]
			else:
				for k in range(3):
					bg[i][j][k] = img[i][j][k]
	return fg, bg

def draw_line(event, x, y, flags, param):
	global drawing, MODE
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if MODE:
				cv2.circle(mask, (x, y), 5, (0), -1)
				cv2.circle(mask_img, (x, y), 5, (0, 0, 0), -1)
			else:
				cv2.circle(mask, (x, y), 5, (255), -1)
				cv2.circle(mask_img, (x, y), 5, (255, 255, 255), -1)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if MODE:
			cv2.circle(mask, (x, y), 5, (0), -1)
			cv2.circle(mask_img, (x, y), 5, (0, 0, 0), -1)
		else:
			cv2.circle(mask, (x, y), 5, (255), -1)
			cv2.circle(mask_img, (x, y), 5, (255, 255, 255), -1)
	else:
		pass

img = cv2.imread("a.jpg")
img = cv2.resize(img, (512, 512))
mask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
for i in range(mask.shape[0]):
	for j in range(mask.shape[1]):
		mask[i][j] = 128
mask_img = img.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback('Image', draw_line)
while True:
	cv2.imshow("Image", mask_img)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		MODE = not MODE
	elif k == 27:
		break
fg, bg = GraphCut(img, mask)
cv2.namedWindow("fg")
cv2.namedWindow("bg")
while True:
	cv2.imshow("fg", fg)
	cv2.imshow("bg", bg)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
cv2.destroyAllWindows()