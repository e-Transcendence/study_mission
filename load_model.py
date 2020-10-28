from gensim.models import Word2Vec

model = Word2Vec.load("D:\All_project\study_mission\\300features_40minwords_10context")
result = []
result.append(model.doesnt_match("man woman child kitchen".split()))
result.append(model.doesnt_match("france england germany berlin".split()))
result.append(model.doesnt_match("paris berlin london austria".split()))
result.append(model.most_similar("man"))
result.append(model.most_similar("queen"))
result.append(model.most_similar("awful"))

for i in result:
    print(i)
