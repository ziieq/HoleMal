"""
whole_dataset
rank_list1 = [
    [5, 1, 1],
    [6, 2, 5],
    [4, 7, 4],
    [1, 4, 3],
    [2, 3, 2],
    [3, 6, 7],
    [7, 5, 6]
]
rank_list2 = [
    [7, 1, 1],
    [6, 2, 5],
    [3, 7, 4],
    [1, 4, 3],
    [2, 3, 2],
    [4, 6, 7],
    [5, 5, 6]
]
rank_list3 = [
    [6, 1, 1],
    [7, 4, 7],
    [1, 7, 4],
    [2, 3, 3],
    [3, 2, 2],
    [4, 6, 6],
    [5, 5, 5]
]

rank_list4 = [
    [7, 1, 1],
    [4, 4, 7],
    [3, 7, 4],
    [1, 3, 2],
    [2, 2, 3],
    [6, 6, 6],
    [5, 5, 5]
]
"""
rank_list1 = [
    [4, 1, 1],
    [5, 2, 4],
    [2, 7, 5],
    [1, 6, 3],
    [3, 3, 2],
    [6, 4, 7],
    [7, 5, 6]
]
rank_list2 = [
    [6, 1, 1],
    [4, 2, 5],
    [2, 7, 4],
    [1, 6, 3],
    [3, 5, 2],
    [5, 3, 7],
    [7, 4, 6]
]
rank_list3 = [
    [6, 3, 1],
    [7, 6, 7],
    [2, 7, 4],
    [1, 5, 3],
    [5, 4, 2],
    [3, 2, 6],
    [4, 1, 5]
]

rank_list4 = [
    [4, 4, 2],
    [5, 2, 4],
    [3, 7, 5],
    [1, 5, 3],
    [2, 1, 1],
    [7, 6, 7],
    [6, 3, 6]
]
def mean(l):
    return sum(l) / len(l)


score_list = []
for row in rank_list4:
    f1_rank = row[0]
    ef_rank = mean(row[1:4])
    score = (f1_rank + ef_rank) / 2
    score_list.append(score)
print(score_list)

"""
score = mean(rank_f1 + mean(rank_memory, rank_predict_time)) 
"""