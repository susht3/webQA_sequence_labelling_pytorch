import sys
sys.path.append('../main')
import numpy as np
from util import split_law

def trace(pre, x):
	ret = []
	if 'val' in x:
		ret.append((pre, x['val']))
	order = []
	for i in x:
		if i == 'val':
			continue
		if 'val' in x[i]:
			order.append((x[i]['val'], i))
		else:
			order.append((0, i))
	order = sorted(order, reverse=True)
	for i in order:
		ret.extend(trace(i[1], x[i[1]]))
	return ret


def reorder(x):
	tree = {}
	for i in x:
		split_res = split_law(str(i[0]))
		now = tree
		for j in split_res:
			if j not in now:
				now[j] = {}
			now = now[j]
		now['val'] = i[1]
	ret = trace('', tree)
	ret = list(filter(lambda x: len(split_law(str(x[0])))>1, ret))
	return ret


def tem_normalise_name(x):
	source = set(['《最高人民法院关于办理妨害信用卡管理刑事案件具体应用法律若干问题的解释》',
	'《最高人民法院、最高人民检察院﹤关于办理妨害信用卡管理刑事案件具体应用法律若干问题的解释﹥》',
	'《最高人民法院最高人民检察院关于办理妨害信用卡管理刑事案件具体应用法律若干问题的解释》'])
	tar = '《最高人民法院、最高人民检察院关于办理妨害信用卡管理刑事案件具体应用法律若干问题的解释》'
	if x in source:
		return tar
	return x


def tem_remove(x):
	return x not in ['《中华人民共和国刑法》第五十二条', '《中华人民共和国刑法》第五十三条']

def pri(text):
	for i in text:
		print(i)

def pred(clf, x):
	#print(x)
	inputs =  list(filter(lambda x:len(x)>0, x.split('。')))
	if len(inputs)>1:
		inputs.extend([x])
	combine_ret = clf.match(inputs)
	print("\nmatch res:")
	pri(combine_ret)
	ret = {}
	for idx,res in enumerate(combine_ret):
		res = list(filter(lambda x:tem_remove(x[0]), res))
		res = reorder(res)
		combine_ret[idx] = (inputs[idx], res)
		for i in res:
			ret[i[0]] = max(ret.get(i[0], 0), i[1])
	ret = ret.items()
	ret = reorder(ret)
	combine_ret.append(('', ret))
	'''
	for i in range(len(combine_ret)):
		text = combine_ret[i][0]
		res = list(map(lambda x:(x[0],str(x[1])), combine_ret[i][1]))
		combine_ret[i] = (text, res)
	#print("\npred res:")
	#pri(combine_ret)
	'''
	return combine_ret


if __name__ == '__main__':
	reorder([('《中华刑法》第三条',123), ('《中华刑法》第二条',226), ('《中华刑法》第三条第四款第五项',226)
		])
