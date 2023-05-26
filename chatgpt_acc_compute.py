result = []
with open('logs/chatgpt/cleaned/high_893.txt', 'r') as f:
    for line in f.readlines():
        qid, pred, gt = line.split()
        result.append((qid, pred, gt))

correct_num = 0
wrong_num = 0
others_num = 0
for x in result:
    qid, pred, gt = x
    if pred==gt:
        correct_num += 1
    elif pred != gt and pred != "X":
        wrong_num += 1
    elif pred == "X":
        others_num += 1

print("All num:", len(result))
print("correct_num", correct_num)
print("wrong num", wrong_num)
print("others_num", others_num)

print("ACC:", correct_num/len(result))