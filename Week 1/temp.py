_ = int(input())
for x in range(_):
    inp = input().split()
    arr = list(inp[0])
    brr = list(inp[1])

    mpp = {}
    for i in brr:
        mpp[i] = mpp.get(i,0) + 1

    n = len(arr)
    for i in range(n-1,-1,-1):
        if mpp.get(arr[i],0)>0:
            mpp[arr[i]] -= 1
        else:
            arr[i] = '.'

    new_list = []
    j = 0
    for i in arr:
        if arr[j] != '.':
            new_list.append(arr[j])
        j+=1
    if new_list == brr:
        print('YES')
    else:
        print('NO')

