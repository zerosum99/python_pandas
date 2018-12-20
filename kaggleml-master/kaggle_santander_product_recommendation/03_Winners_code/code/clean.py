# 훈련 데이터와 테스트 데이터를 하나의 데이터로 통합하는 코드이다.
def clean_data(fi, fo, header, suffix):
    
    # fi : 훈련/테스트 데이터를 읽어오는 file iterator
    # fo : 통합되는 데이터가 write되는 경로
    # header : 데이터에 header 줄을 추가할 것인지를 결정하는 boolean
    # suffix : 훈련 데이터에는 48개의 변수가 있고, 테스트 데이터에는 24개의 변수만 있다. suffix로 부족한 테스트 데이터 24개분을 공백으로 채운다.

    # csv의 첫줄, 즉 header를 읽어온다
    head = fi.readline().strip("\n").split(",")
    head = [h.strip('"') for h in head]

    # ‘nomprov’ 변수의 위치를 ip에 저장한다
    for i, h in enumerate(head):
        if h == "nomprov":
            ip = i

    # header가 True 일 경우에는, 저장할 파일의 header를 write한다
    if header:
        fo.write("%s\n" % ",".join(head))

    # n은 읽어온 변수의 개수를 의미한다 (훈련 데이터 : 48, 테스트 데이터 : 24)
    n = len(head)
    for line in fi:
        # 파일의 내용을 한줄 씩 읽어와서, 줄바꿈(\n)과 ‘,’으로 분리한다
        fields = line.strip("\n").split(",")

        # ‘nomprov’변수에 ‘,’을 포함하는 데이터가 존재한다. ‘,’으로 분리된 데이터를 다시 조합한다
        if len(fields) > n:
            prov = fields[ip] + fields[ip+1]
            del fields[ip]
            fields[ip] = prov

        # 데이터 개수가 n개와 동일한지 확인하고, 파일에 write한다. 테스트 데이터의 경우, suffix는24개의 공백이다
        assert len(fields) == n
        fields = [field.strip() for field in fields]
        fo.write("%s%s\n" % (",".join(fields), suffix))

# 하나의 데이터로 통합하는 코드를 실행한다. 먼저 훈련 데이터를 write하고, 그 다음으로 테스트 데이터를 write한다. 이제부터 하나의 dataframe만을 다루며 데이터 전처리를 진행한다.
with open("../input/8th.clean.all.csv", "w") as f:
    clean_data(open("../input/train_ver2.csv"), f, True, "")
    comma24 = "".join(["," for i in range(24)])
    clean_data(open("../input/test_ver2.csv"), f, False, comma24)
