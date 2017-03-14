whole = []

with open("real_data.txt") as f:
    for item in f:
        item = item.strip().split()
        parse_line = [int(x) for x in item]
        whole.extend(parse_line)
f.close()
print len(set(whole))
