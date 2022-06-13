import numpy as np

'''
The key of creating high differentiable visualization is constrain the pixel value display range.
In QuPath, 1% dark pixels and 1% bright pixels should be saturated. Use this criteria to calculate the lower and upper 
bound of display range within the histogram (n_bin=1024) of image array.
'''

class histogram_MxIF:
    def __init__(self, img_arr, NUM_BINS = 1024):
        self.hist, self.bin_edges = np.histogram(img_arr, bins=NUM_BINS)

    def getCountSum(self):
        return sum(self.hist)

    def nBins(self):
        return len(self.hist)

    def getCountsForBin(self, idx):
        return self.hist[idx]

    def getEdgeMin(self):
        return self.bin_edges[0]

    def getEdgeMax(self):
        return self.bin_edges[-1]

    def getBinLeftEdge(self, idx):
        return self.bin_edges[idx]

    def getBinRightEdge(self, idx):
        return self.bin_edges[idx+1]

    def getBinWidth(self, ind):
        return self.getBinRightEdge(ind) - self.getBinLeftEdge(ind)

# set range of display
def autoSetDisplayRange(img_arr, saturation=0.01):
    '''
    get the min and max pixel value in the display
    :param img_arr: image array
    :param saturation: 	 Controls percentage of saturated pixels to apply when automatically setting brightness/contrast.
	 * A value of 0.01 indicates that approximately 1% dark pixels and 1% bright pixels should be saturated.
    :return:
    '''
    histogram = histogram_MxIF(img_arr)
    if saturation <= 0 or saturation >= 1:
        print("Cannot set display range, wrong saturation")

    countSum = histogram.getCountSum()
    nBins = histogram.nBins()
    ind = 0
    # Possibly skip the first and/or last bins; these can often represent unscanned/clipped regions
    if nBins > 2:
        firstCount = histogram.getCountsForBin(0)
        if firstCount > histogram.getCountsForBin(1):
            countSum -= histogram.getCountsForBin(0)
            ind = 1
        lastCount = histogram.getCountsForBin(nBins - 1)
        if lastCount > histogram.getCountsForBin(nBins - 2):
            countSum -= lastCount
            nBins -= 1

    countMax = countSum * saturation
    count = countMax
    minDisplay = histogram.getEdgeMin()
    while ind < histogram.nBins():
        nextCount = histogram.getCountsForBin(ind)
        if count < nextCount:
            minDisplay = histogram.getBinLeftEdge(ind) + (count / nextCount) * histogram.getBinWidth(ind)
            break
        count -= nextCount
        ind += 1

    count = countMax
    maxDisplay = histogram.getEdgeMax()
    ind = histogram.nBins() - 1
    while ind >= 0:
        nextCount = histogram.getCountsForBin(ind)
        if count < nextCount:
            maxDisplay = histogram.getBinRightEdge(ind) - (count / nextCount) * histogram.getBinWidth(ind)
            break
        count -= nextCount
        ind -= 1
    return minDisplay, maxDisplay

