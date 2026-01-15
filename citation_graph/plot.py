import matplotlib.pyplot as plt

# Data from your output
citations = [1,2,3,4,5,6,7,8,9,10]
articles  = [248803,118007,70462,47143,33992,25743,20193,16344,13675,11548]

plt.figure(figsize=(10,5))
plt.bar(citations, articles)
plt.xticks(citations)
plt.xlabel("Number of citations")
plt.ylabel("Number of papers")
plt.title("Citation distribution (first 10 values)")
plt.tight_layout()
plt.show()
