import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

val path = "hdfs:///user/ubuntu/arxiv_data/edges/arxiv_edges.csv"

val rawEdges: RDD[(String, String)] =
  sc.textFile(path)
    .filter(line => !line.startsWith("src,"))
    .flatMap { line =>
      val i = line.indexOf(',')
      if (i < 0) None else Some((line.substring(0, i), line.substring(i + 1)))
    }
    .persist(StorageLevel.MEMORY_AND_DISK)
// val rawEdgesAll: RDD[(String, String)] =
//   sc.textFile(path)
//     .filter(line => !line.startsWith("src,"))
//     .flatMap { line =>
//       val i = line.indexOf(',')
//       if (i < 0) None else Some((line.substring(0, i), line.substring(i + 1)))
//     }

// val rawEdges: RDD[(String, String)] =
//   rawEdgesAll
//     .sample(withReplacement = false, fraction = 0.3, seed = 42L)
//     .persist(StorageLevel.MEMORY_AND_DISK)


val ids: RDD[String] =
  rawEdges.flatMap { case (s, d) => Iterator(s, d) }.distinct()
    .persist(StorageLevel.MEMORY_AND_DISK)

val id2vid: RDD[(String, VertexId)] =
  ids.zipWithUniqueId().map { case (id, vid) => (id, vid) }
    .persist(StorageLevel.MEMORY_AND_DISK)

val vertices: RDD[(VertexId, String)] =
  id2vid.map { case (id, vid) => (vid, id) }

val withSrcVid: RDD[(String, (String, VertexId))] =
  rawEdges.join(id2vid) // key=srcStr => (dstStr, srcVid)

val edges: RDD[Edge[Int]] =
  withSrcVid
    .map { case (_, (dstStr, srcVid)) => (dstStr, srcVid) }
    .join(id2vid)
    .map { case (_, (srcVid, dstVid)) => Edge(srcVid, dstVid, 1) }
    .persist(StorageLevel.MEMORY_AND_DISK)

val graph: Graph[String, Int] = Graph(vertices, edges, defaultVertexAttr = "")

// val graph: Graph[String, Int] = Graph(vertices, edges, defaultVertexAttr = "").persist(StorageLevel.MEMORY_AND_DISK)

import org.apache.spark.graphx.lib.LabelPropagation

val edgesClean: RDD[Edge[Int]] =
  edges
    .filter(e => e.srcId != e.dstId)
    .map(e => ((e.srcId, e.dstId), 1))
    .reduceByKey((_, _) => 1)
    .map { case ((s, d), _) => Edge(s, d, 1) }
    .persist(StorageLevel.MEMORY_AND_DISK)

val directedGraph: Graph[String, Int] =
  Graph(vertices, edgesClean, defaultVertexAttr = "")

val edgesUndir: RDD[Edge[Int]] =
  edgesClean
    .flatMap(e => Iterator(e, Edge(e.dstId, e.srcId, e.attr)))
    .filter(e => e.srcId != e.dstId)
    .map(e => ((e.srcId, e.dstId), 1))
    .reduceByKey((_, _) => 1)
    .map { case ((s, d), _) => Edge(s, d, 1) }
    .persist(StorageLevel.MEMORY_AND_DISK)

val undirectedGraph: Graph[String, Int] =
  Graph(vertices, edgesUndir, defaultVertexAttr = "")

println(s"numVertices = ${directedGraph.numVertices}")
println(s"numEdges    = ${directedGraph.numEdges}")

val topOut = directedGraph.outDegrees.join(vertices).map { case (_, (deg, id)) => (deg, id) }
  .top(20)(Ordering.by(_._1))
println("Top 20 out-degree:")
topOut.foreach { case (deg, id) => println(s"$deg\t$id") }

val topIn = directedGraph.inDegrees.join(vertices).map { case (_, (deg, id)) => (deg, id) }
  .top(20)(Ordering.by(_._1))
println("Top 20 in-degree:")
topIn.foreach { case (deg, id) => println(s"$deg\t$id") }

val cc = directedGraph.connectedComponents().vertices
val ccSizes = cc.map { case (_, comp) => (comp, 1L) }.reduceByKey(_ + _)
println("Top 10 connected components by size:")
ccSizes.map { case (comp, n) => (n, comp) }.top(10)(Ordering.by(_._1))
  .foreach { case (n, comp) => println(s"$n\t$comp") }

val pr = directedGraph.pageRank(1e-4).vertices
val topPR = pr.join(vertices).map { case (_, (rank, id)) => (rank, id) }
  .top(10)(Ordering.by(_._1))
println("Top 10 PageRank:")
topPR.foreach { case (rank, id) => println(f"$rank%.6f\t$id") }

println("\=== Advanced Analysis: Communities and Detailed Statistics ===\n")



val triCounts = undirectedGraph.triangleCount().vertices

val topTriangles = triCounts.join(vertices)
  .map { case (vid, (count, idStr)) => (count, idStr) }
  .top(5)(Ordering.by(_._1))

println("Top 5 articles involved in triangles (community cores):")
topTriangles.foreach { case (count, idStr) => println(s"Triangles: $count \t ID: $idStr") }


import org.apache.spark.graphx.lib.LabelPropagation

val lpaGraph = LabelPropagation.run(undirectedGraph, 5)

val communities = lpaGraph.vertices.map { case (vid, label) => (label, 1L) }
  .reduceByKey(_ + _)
  .sortBy(_._2, ascending = false)

println("\nTop 10 Largest Detected Communities (Label Propagation):")
communities.take(10).foreach { case (label, count) => 
  println(s"Community ID: $label \t Number of papers: $count") 
}

val biggestCommunityLabel = communities.first()._1
val papersInBiggestComm = lpaGraph.vertices
  .filter { case (vid, label) => label == biggestCommunityLabel }
  .join(vertices)
  .take(5)

println(s"\nExamples of papers from the largest community (ID $biggestCommunityLabel):")
papersInBiggestComm.foreach { case (vid, (label, idStr)) => println(s"- $idStr") }

val inDegreeDistribution = directedGraph.inDegrees
  .map { case (_, deg) => (deg, 1L) }
  .reduceByKey(_ + _)
  .sortByKey()

println("\nCitation distribution (first 10 values):")
println("(Number of citations -> How many papers have this number)")
inDegreeDistribution.take(10).foreach { case (deg, count) => println(s"$deg citation -> $count papers") }

val totalInDegree = directedGraph.inDegrees.map(_._2).reduce(_ + _)
val countInDegreeNodes = directedGraph.inDegrees.count()
val avgInDegree = totalInDegree.toDouble / countInDegreeNodes

println(f"\nAverage citations per paper (for papers cited at least once): $avgInDegree%.2f")

println("\n=== FINAL STAGE: COMPLEX ANALYSES & EXPORT ===")



import org.apache.spark.graphx.EdgeDirection

println("\n--- Degrees of Separation (BFS Pregel, nedirecționat) ---")

val (sourceVid, sourceDeg) =
  directedGraph.inDegrees.top(1)(Ordering.by(_._2)).head

val sourceIdStr = directedGraph.vertices.lookup(sourceVid).headOption.getOrElse(sourceVid.toString)
println(s"Source (most cited): $sourceIdStr (inDegree=$sourceDeg)")

val INF = Int.MaxValue / 4

val bfsInit = undirectedGraph.mapVertices { case (vid, _) =>
  if (vid == sourceVid) 0 else INF
}.cache()


val bfs = bfsInit.pregel(INF, activeDirection = EdgeDirection.Either)(
  (vid, dist, newDist) => math.min(dist, newDist),

  triplet => {
    val srcD = triplet.srcAttr
    val dstD = triplet.dstAttr
    val msgs = scala.collection.mutable.ArrayBuffer.empty[(VertexId, Int)]

    if (srcD + 1 < dstD) msgs += ((triplet.dstId, srcD + 1))
    if (dstD + 1 < srcD) msgs += ((triplet.srcId, dstD + 1)) // pentru nedirecționat

    msgs.iterator
  },

  (a, b) => math.min(a, b)
).cache()

val dists = bfs.vertices.map(_._2).filter(_ < INF).cache()

println(s"Reachable (undirected) from the source: ${dists.count()} noduri")
println(s"Max hops (approximate diameter from the source): ${dists.max()}")
println(f"Mean hops: ${dists.map(_.toDouble).mean()}%.3f")

val hist = dists.map(d => (d, 1L)).reduceByKey(_ + _).sortByKey()
println("Distance histogram (hops -> number of papers), first 30:")
hist.take(30).foreach { case (d, c) => println(s"$d -> $c") }




def saveCsvSingleFile(outDir: String, header: String, lines: RDD[String]): Unit = {
  sc.parallelize(Seq(header)).union(lines).coalesce(1).saveAsTextFile(outDir)
}

def csvQ(s: String): String = {
  val safe = Option(s).getOrElse("").replace("\"", "\"\"")
  "\"" + safe + "\""
}


val gephiBaseDir = "hdfs:///user/ubuntu/arxiv_data/gephi_exports"


val TOP_N = 20         
val TOP_N_BRIDGES = TOP_N * 2  
val K_CORE = 3           



println(s"\n--- Export Gephi: (1) Community map Top $TOP_N by PageRank ---")


val idByVid: RDD[(VertexId, String)] = vertices // (vid -> arxivId)
val commByVid: VertexRDD[VertexId] = lpaGraph.vertices // (vid -> communityLabel)
val inDegAll = directedGraph.inDegrees
val outDegAll = directedGraph.outDegrees


val topNArr = pr.map { case (vid, rank) => (rank, vid) }
  .top(TOP_N)(Ordering.by(_._1))

val topNVids = topNArr.map(_._2).toSet
val topNVidsBC = sc.broadcast(topNVids)

val topNRDD: RDD[(VertexId, Double)] =
  sc.parallelize(topNArr).map { case (rank, vid) => (vid, rank) } // (vid, pr)

val edgesTopNDirected: RDD[Edge[Int]] =
  directedGraph.edges
    .filter(e => topNVidsBC.value.contains(e.srcId) && topNVidsBC.value.contains(e.dstId))
    .map(e => Edge(e.srcId, e.dstId, 1))
    .persist(StorageLevel.MEMORY_AND_DISK)

val nodesTopNLines: RDD[String] =
  topNRDD
    .leftOuterJoin(idByVid) // (vid, (pr, idOpt))
    .mapValues { case (rank, idOpt) => (rank, idOpt.getOrElse("")) }
    .leftOuterJoin(inDegAll)
    .mapValues { case ((rank, id), inOpt) => (rank, id, inOpt.getOrElse(0)) }
    .leftOuterJoin(outDegAll)
    .mapValues { case ((rank, id, inD), outOpt) => (rank, id, inD, outOpt.getOrElse(0)) }
    .leftOuterJoin(commByVid)
    .map {case (vid, ((rank, id0, inD, outD), commOpt)) =>
      val id = if (id0 != null && id0.nonEmpty) id0 else vid.toString  // fallback dacă lipsește
      val comm = commOpt.getOrElse(-1L)
      val url = s"https://arxiv.org/abs/$id"

      
      val vizLabel = s"$id (in=$inD)"

      
      s"$vid,${csvQ(vizLabel)},${csvQ(id)},$rank,$inD,$outD,$comm,$url"

    }

val edgesTopNLines: RDD[String] =
  edgesTopNDirected.map(e => s"${e.srcId},${e.dstId},Directed")

val outCommunityNodes = s"$gephiBaseDir/community_top${TOP_N}/nodes_csv"
val outCommunityEdges = s"$gephiBaseDir/community_top${TOP_N}/edges_csv"

saveCsvSingleFile(outCommunityNodes, "Id,Label,PageRank,InDegree,OutDegree,Community,VizLabel,Url", nodesTopNLines)
saveCsvSingleFile(outCommunityEdges, "Source,Target,Type", edgesTopNLines)

println(s"Saved (community map) nodes: $outCommunityNodes")
println(s"Saved (community map) edges: $outCommunityEdges")


println(s"\n--- Export Gephi: (2) Backbone k-core=$K_CORE on Top $TOP_N PageRank ---")

val edgesTopNUndir: RDD[Edge[Int]] =
  edgesTopNDirected
    .flatMap(e => Iterator(e, Edge(e.dstId, e.srcId, 1)))
    .map(e => ((e.srcId, e.dstId), 1))
    .reduceByKey((_, _) => 1)
    .map { case ((s, d), _) => Edge(s, d, 1) }
    .persist(StorageLevel.MEMORY_AND_DISK)

val topNVertexAttrs: RDD[(VertexId, String)] =
  sc.parallelize(topNVids.toSeq).map(vid => (vid, "")) // placeholder
    .leftOuterJoin(idByVid)
    .map { case (vid, (_, idOpt)) => (vid, idOpt.getOrElse("")) }

var coreG: Graph[(String, Boolean), Int] =
  Graph(topNVertexAttrs.map { case (vid, id) => (vid, (id, true)) }, edgesTopNUndir, defaultVertexAttr = ("", true))
    .cache()

var prevN = -1L
var iter = 0
val MAX_ITERS = 30

while (iter < MAX_ITERS && coreG.numVertices != prevN) {
  prevN = coreG.numVertices
  val deg = coreG.degrees // (vid -> degree)
  val marked = coreG.outerJoinVertices(deg) { (vid, attr, degOpt) =>
    val d = degOpt.getOrElse(0)
    (attr._1, d >= K_CORE)
  }
  val pruned = marked.subgraph(vpred = (vid, attr) => attr._2).cache()
  coreG.unpersistVertices(blocking = false)
  coreG.edges.unpersist(blocking = false)
  coreG = pruned.mapVertices { case (vid, attr) => (attr._1, true) }.cache()
  iter += 1
}

val coreVids = coreG.vertices.map { case (vid, (id, _)) => vid }.collect().toSet
val coreVidsBC = sc.broadcast(coreVids)

println(s"Backbone k-core=$K_CORE vertices: ${coreVids.size} (from TOP_N=$TOP_N)")

val edgesBackboneDirected: RDD[Edge[Int]] =
  directedGraph.edges
    .filter(e => coreVidsBC.value.contains(e.srcId) && coreVidsBC.value.contains(e.dstId))
    .map(e => Edge(e.srcId, e.dstId, 1))
    .persist(StorageLevel.MEMORY_AND_DISK)

val backboneGraph = Graph(coreG.vertices.map { case (vid, (id, _)) => (vid, id) }, edgesBackboneDirected, defaultVertexAttr = "")
val inDegBackbone = backboneGraph.inDegrees
val outDegBackbone = backboneGraph.outDegrees

val backboneNodesLines: RDD[String] =
  backboneGraph.vertices
    .map { case (vid, id) => (vid, id) }
    .leftOuterJoin(pr) // (vid, (id, prOpt))
    .mapValues { case (id, prOpt) => (id, prOpt.getOrElse(0.0)) }
    .leftOuterJoin(inDegBackbone)
    .mapValues { case ((id, prVal), inOpt) => (id, prVal, inOpt.getOrElse(0)) }
    .leftOuterJoin(outDegBackbone)
    .mapValues { case ((id, prVal, inD), outOpt) => (id, prVal, inD, outOpt.getOrElse(0)) }
    .leftOuterJoin(commByVid)
    .map { case (vid, ((id0, prVal, inD, outD), commOpt)) =>
      val id = if (id0 != null && id0.nonEmpty) id0 else vid.toString
      val comm = commOpt.getOrElse(-1L)
      val url = s"https://arxiv.org/abs/$id"
      val vizLabel = s"$id (in=$inD)"

      s"$vid,${csvQ(vizLabel)},${csvQ(id)},$prVal,$inD,$outD,$comm,$url"

    }

val backboneEdgesLines: RDD[String] =
  edgesBackboneDirected.map(e => s"${e.srcId},${e.dstId},Directed")

val outBackboneNodes = s"$gephiBaseDir/backbone_k${K_CORE}_top${TOP_N}/nodes_csv"
val outBackboneEdges = s"$gephiBaseDir/backbone_k${K_CORE}_top${TOP_N}/edges_csv"

saveCsvSingleFile(outBackboneNodes, "Id,Label,PageRank,InDegree,OutDegree,Community,VizLabel,Url", backboneNodesLines)
saveCsvSingleFile(outBackboneEdges, "Source,Target,Type", backboneEdgesLines)

println(s"Saved (backbone) nodes: $outBackboneNodes")
println(s"Saved (backbone) edges: $outBackboneEdges")


println(s"\n--- Export Gephi: (3) Bridges/Brokers Top $TOP_N_BRIDGES by PageRank ---")

val topBrrArr = pr.map { case (vid, rank) => (rank, vid) }
  .top(TOP_N_BRIDGES)(Ordering.by(_._1))

val topBrrVids = topBrrArr.map(_._2).toSet
val topBrrVidsBC = sc.broadcast(topBrrVids)

val topBrrRDD: RDD[(VertexId, Double)] =
  sc.parallelize(topBrrArr).map { case (rank, vid) => (vid, rank) } // (vid, pr)

val edgesBrrDirected: RDD[Edge[Int]] =
  directedGraph.edges
    .filter(e => topBrrVidsBC.value.contains(e.srcId) && topBrrVidsBC.value.contains(e.dstId))
    .map(e => Edge(e.srcId, e.dstId, 1))
    .persist(StorageLevel.MEMORY_AND_DISK)

val edgesBrrUndirPairs: RDD[(VertexId, VertexId)] =
  edgesBrrDirected.flatMap(e => Iterator((e.srcId, e.dstId), (e.dstId, e.srcId)))

val neighborCommPairs: RDD[(VertexId, VertexId)] =
  edgesBrrUndirPairs.map { case (vid, nvid) => (nvid, vid) } 
    .join(commByVid)                                        
    .map { case (_, (vid, commN)) => (vid, commN) }         
    .distinct()

val neighborCommCount: RDD[(VertexId, Int)] =
  neighborCommPairs
    .map { case (vid, _) => (vid, 1) }
    .reduceByKey(_ + _)

val bridgeScore: RDD[(VertexId, Double)] =
  neighborCommCount
    .mapValues(_.toDouble)
    .join(topBrrRDD)
    .mapValues { case (nComm, prVal) => nComm * prVal }

val bridgesNodesLines: RDD[String] =
  topBrrRDD
    .leftOuterJoin(idByVid)
    .mapValues { case (rank, idOpt) => (rank, idOpt.getOrElse("")) }
    .leftOuterJoin(inDegAll)
    .mapValues { case ((rank, id), inOpt) => (rank, id, inOpt.getOrElse(0)) }
    .leftOuterJoin(outDegAll)
    .mapValues { case ((rank, id, inD), outOpt) => (rank, id, inD, outOpt.getOrElse(0)) }
    .leftOuterJoin(commByVid)
    .mapValues { case ((rank, id, inD, outD), commOpt) => (rank, id, inD, outD, commOpt.getOrElse(-1L)) }
    .leftOuterJoin(neighborCommCount)
    .mapValues { case ((rank, id, inD, outD, comm), nCommOpt) => (rank, id, inD, outD, comm, nCommOpt.getOrElse(0)) }
    .leftOuterJoin(bridgeScore)
    .map { case (vid, ((rank, id0, inD, outD, comm, nComm), bScoreOpt)) =>
      val id = if (id0 != null && id0.nonEmpty) id0 else vid.toString
      val bScore = bScoreOpt.getOrElse(0.0)
      val url = s"https://arxiv.org/abs/$id"

      val vizLabel = f"$id (nComm=$nComm, bScore=$bScore%.3f)"

      s"$vid,${csvQ(vizLabel)},${csvQ(id)},$rank,$inD,$outD,$comm,$nComm,$bScore,$url"

    }

val bridgesEdgesLines: RDD[String] =
  edgesBrrDirected.map(e => s"${e.srcId},${e.dstId},Directed")

val outBridgesNodes = s"$gephiBaseDir/bridges_top${TOP_N_BRIDGES}/nodes_csv"
val outBridgesEdges = s"$gephiBaseDir/bridges_top${TOP_N_BRIDGES}/edges_csv"

saveCsvSingleFile(outBridgesNodes, "Id,Label,PageRank,InDegree,OutDegree,Community,NeighborCommunities,BridgeScore,VizLabel,Url", bridgesNodesLines)
saveCsvSingleFile(outBridgesEdges, "Source,Target,Type", bridgesEdgesLines)

println(s"Saved (bridges) nodes: $outBridgesNodes")
println(s"Saved (bridges) edges: $outBridgesEdges")

val topBridgeCandidates = bridgeScore
  .join(idByVid) 
  .map { case (_, (score, id)) => (score, id) }
  .top(20)(Ordering.by(_._1))

println("\nTop 20 bridge candidates (BridgeScore = NeighborCommunities * PageRank):")
topBridgeCandidates.foreach { case (s, id) => println(f"$s%.6f\t$id") }


topNVidsBC.destroy()
coreVidsBC.destroy()
topBrrVidsBC.destroy()

println("\n--- Gephi import hints ---")
println(s"(1) Community map: import nodes from $outCommunityNodes/part-* and edges from $outCommunityEdges/part-*")
println(s"(2) Backbone: import nodes from $outBackboneNodes/part-* and edges from $outBackboneEdges/part-*")
println(s"(3) Bridges: import nodes from $outBridgesNodes/part-* and edges from $outBridgesEdges/part-*")
println("In Gephi: Appearance->Nodes: Size=PageRank (sau InDegree), Color=Community; Layout=ForceAtlas2 + Prevent overlap.")
println("For Bridges: runs Betweenness Centrality in Gephi on the dataset (3) and sets Size=Betweenness, Color=Community.")





println("\nAnalysis completed.")

System.exit(0)
