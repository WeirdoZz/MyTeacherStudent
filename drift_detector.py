import numpy as np
import scipy.stats


class HypothesisTestDetector(object):
    METHOD = "tt"
    def __init__(self,method,window,thr):
        """
        :param method: str, 比较概率分布的方法,必须是["ks","wrs","tt"]中的
        :param window: int ,用于比较分布的数据窗口大小
        :param thr: 两个分布的差异的阈值，大于阈值则判断为检测到漂移
        """
        assert method in ["ks","wrs","tt"]

        if method =="ks":
            m=scipy.stats.ks_2samp
        elif method =="wrs":
            m=scipy.stats.ranksums
        else:
            m=scipy.stats.ttest_ind

        self.method=m
        self.alarm_list=[]
        self.data=[]
        self.window=window
        self.thr=thr
        self.index=0

    def add_element(self,elem):
        self.data.append(elem)

    def detected_change(self):
        x=np.array(self.data)
        w=self.window

        if len(x)<2*w:
            self.index+=1
            return False

        testw=x[-w:]
        refw=x[-(w*2):-w]

        ht=self.method(testw,refw)
        pval=ht[1]
        has_change=pval<self.thr

        if has_change:
            print(f"在{self.index}处发生了漂移")
            self.alarm_list.append(self.index)
            self.index+=1
            # 如果发生漂移，之前的数据就没有用了
            self.data=list(x[-w,:])
            return True
        else:
            self.index+=1
            return False
