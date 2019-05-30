PATH = '/Users/gglee/Develop/tensorflow/tensorflow/lite/examples/android/app/src/main/assets/box_priors.txt'

with open(PATH) as rf:
    for i, l in enumerate(rf.readlines()):
        ll = l.split()
        print(i, len(ll))
        print(l[0:100])

        # 4 x 1917 ( = 4 x 213x3x3 )
        # 5 aspect ratio?
