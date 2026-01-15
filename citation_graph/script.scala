import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.graphx.lib.LabelPropagation
import org.apache.spark.graphx.EdgeDirection

import org.apache.spark.sql.functions._
import org.apache.hadoop.fs.{FileSystem, Path}


val path = "hdfs:///user/ubuntu/arxiv_data/edges/arxiv_edges.csv"
val kaggle_metadata = "hdfs:///user/ubuntu/arxiv_data/arxiv-metadata-oai-snapshot.json"



val outHtmlPathStr = "hdfs:///user/ubuntu/arxiv_data/results/graphx_results_all.html"


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

// build id2vid + vertices
val ids: RDD[String] =
  rawEdges.flatMap { case (s, d) => Iterator(s, d) }
    .distinct()
    .persist(StorageLevel.MEMORY_AND_DISK)

val id2vid: RDD[(String, VertexId)] =
  ids.zipWithUniqueId()
    .map { case (id, vid) => (id, vid) }
    .persist(StorageLevel.MEMORY_AND_DISK)

val vertices: RDD[(VertexId, String)] =
  id2vid.map { case (id, vid) => (vid, id) }

// build directed edges
val withSrcVid: RDD[(String, (String, VertexId))] =
  rawEdges.join(id2vid) 

val edges: RDD[Edge[Int]] =
  withSrcVid
    .map { case (_, (dstStr, srcVid)) => (dstStr, srcVid) }
    .join(id2vid)
    .map { case (_, (srcVid, dstVid)) => Edge(srcVid, dstVid, 1) }
    .persist(StorageLevel.MEMORY_AND_DISK)


// clean edges 
val edgesClean: RDD[Edge[Int]] =
  edges
    .filter(e => e.srcId != e.dstId)
    .map(e => ((e.srcId, e.dstId), 1))
    .reduceByKey((_, _) => 1)
    .map { case ((s, d), _) => Edge(s, d, 1) }
    .persist(StorageLevel.MEMORY_AND_DISK)

// directed graph (citations)
val directedGraph: Graph[String, Int] =
  Graph(vertices, edgesClean, defaultVertexAttr = "")

// undirected (symmetrized) for triangleCount + LPA + BFS 
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


// analyses
val numV = directedGraph.numVertices
val numE = directedGraph.numEdges
println(s"numVertices = $numV")
println(s"numEdges    = $numE")

// top out-degree / in-degree
val topOut: Array[(Int, String)] =
  directedGraph.outDegrees.join(vertices).map { case (_, (deg, id)) => (deg, id) }
    .top(20)(Ordering.by(_._1))

println("Top 20 out-degree:")
topOut.foreach { case (deg, id) => println(s"$deg\t$id") }

val topIn: Array[(Int, String)] =
  directedGraph.inDegrees.join(vertices).map { case (_, (deg, id)) => (deg, id) }
    .top(20)(Ordering.by(_._1))

println("Top 20 in-degree:")
topIn.foreach { case (deg, id) => println(s"$deg\t$id") }

// connected components (on directedGraph; graphx returns wcc-like)
val cc = directedGraph.connectedComponents().vertices
val ccSizes = cc.map { case (_, comp) => (comp, 1L) }.reduceByKey(_ + _)

val topCC: Array[(Long, VertexId)] =
  ccSizes.map { case (comp, n) => (n, comp) }
    .top(10)(Ordering.by(_._1))

println("Top 10 connected components by size:")
topCC.foreach { case (n, comp) => println(s"$n\t$comp") }

// page rank
val pr = directedGraph.pageRank(1e-4).vertices
val topPR: Array[(Double, String)] =
  pr.join(vertices).map { case (_, (rank, id)) => (rank, id) }
    .top(10)(Ordering.by(_._1))

println("Top 10 PageRank:")
topPR.foreach { case (rank, id) => println(f"$rank%.6f\t$id") }

println("\n=== Advanced Analysis: Communities and Detailed Statistics ===\n")

// triangle count on undirected
val triCounts = undirectedGraph.triangleCount().vertices
val topTriangles: Array[(Int, String)] =
  triCounts.join(vertices)
    .map { case (_, (count, idStr)) => (count, idStr) }
    .top(5)(Ordering.by(_._1))

println("Top 5 articles involved in triangles (community cores):")
topTriangles.foreach { case (count, idStr) => println(s"Triangles: $count \t ID: $idStr") }

// label propagation
val lpaGraph = LabelPropagation.run(undirectedGraph, 5)

val communities = lpaGraph.vertices.map { case (_, label) => (label, 1L) }
  .reduceByKey(_ + _)
  .sortBy(_._2, ascending = false)

val topCommunities = communities.take(10)
println("\nTop 10 Largest Detected Communities (Label Propagation):")
topCommunities.foreach { case (label, count) =>
  println(s"Community ID: $label \t Number of papers: $count")
}

val biggestCommunityLabel = topCommunities.head._1
val papersInBiggestCommArr =
  lpaGraph.vertices
    .filter { case (_, label) => label == biggestCommunityLabel }
    .join(vertices)
    .map { case (_, (_, idStr)) => idStr }
    .take(5)

println(s"\nExamples of papers from the largest community (ID $biggestCommunityLabel):")
papersInBiggestCommArr.foreach(idStr => println(s"- $idStr"))

// in-degree distribution (first 10) + avg
val inDegreeDistribution = directedGraph.inDegrees
  .map { case (_, deg) => (deg, 1L) }
  .reduceByKey(_ + _)
  .sortByKey()

val inDegFirst10 = inDegreeDistribution.take(10)

println("\nCitation distribution (first 10 values):")
println("(Number of citations -> How many papers have this number)")
inDegFirst10.foreach { case (deg, count) => println(s"$deg citation -> $count articles") }

val totalInDegree = directedGraph.inDegrees.map(_._2).reduce(_ + _)
val countInDegreeNodes = directedGraph.inDegrees.count()
val avgInDegree = totalInDegree.toDouble / countInDegreeNodes
println(f"\nAverage citations per paper (for papers cited at least once): $avgInDegree%.2f")

println("\n=== FINAL STAGE: COMPLEX ANALYSES & EXPORT ===")

// bfs from top in-degree paper
println("\n--- Degrees of Separation (BFS Pregel, undirected) ---")

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
    if (dstD + 1 < srcD) msgs += ((triplet.srcId, dstD + 1))
    msgs.iterator
  },
  (a, b) => math.min(a, b)
).cache()

val dists = bfs.vertices.map(_._2).filter(_ < INF).cache()

val reachableCount = dists.count()
val maxHops = dists.max()
val meanHops = dists.map(_.toDouble).mean()

println(s"Reachable (undirected) from the source: $reachableCount nodes")
println(s"Max hops (approximate diameter from the source): $maxHops")
println(f"Mean hops: $meanHops%.3f")

val hist = dists.map(d => (d, 1L)).reduceByKey(_ + _).sortByKey()
val hist30 = hist.take(30)

println("Distance histogram (hops -> number of papers), first 30:")
hist30.foreach { case (d, c) => println(s"$d -> $c") }


// html
// helpers
def absUrl(id: String): String = s"https://arxiv.org/abs/$id"
def esc(s: String): String =
  if (s == null) "" else s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


val idsForMeta: Seq[String] =
  (topOut.map(_._2).toSeq ++
   topIn.map(_._2).toSeq ++
   topPR.map(_._2).toSeq ++
   topTriangles.map(_._2).toSeq ++
   papersInBiggestCommArr.toSeq ++
   Seq(sourceIdStr)
  ).distinct


val categoryMapPath = "hdfs:///user/ubuntu/arxiv_data/arxiv_category_map_1.csv"

// csv columns: category_id, pretty_name, group_name
val catMap: Map[String, String] =
  spark.read.option("header", "true").csv(categoryMapPath)
    .select(col("category_id"), col("pretty_name"))
    .na.drop()
    .collect()
    .map(r => r.getString(0) -> r.getString(1))
    .toMap

val catMapBC = sc.broadcast(catMap)


val prettyCategoriesUdf = udf { cats: String =>
  if (cats == null) null
  else {
    cats.trim.split("\\s+").toSeq
      .filter(_.nonEmpty)
      .map(code => catMapBC.value.getOrElse(code, code)) // fallback = original code
      .distinct
      .mkString("; ")
  }
}


val meta = spark.read.json(kaggle_metadata)
  .select(
    col("id").as("arxiv_id"),
    col("title"),
    col("authors"),
    prettyCategoriesUdf(col("categories")).as("categories"),
    col("update_date")
  )
  .withColumn("year",
    when(col("update_date").isNotNull && length(col("update_date")) >= 4, substring(col("update_date"), 1, 4))
      .otherwise(lit(null))
  )

val idsDF = spark.createDataset(idsForMeta).toDF("arxiv_id")

val metaSmall = meta
  .join(broadcast(idsDF), Seq("arxiv_id"), "right")
  .select("arxiv_id", "title", "authors", "categories", "year")

val metaMap: Map[String, (String, String, String, String)] =
  metaSmall.collect().map { r =>
    val id = r.getAs[String]("arxiv_id")
    val title = r.getAs[String]("title")
    val authors = r.getAs[String]("authors")
    val cats = r.getAs[String]("categories")
    val year = r.getAs[String]("year")
    id -> (title, authors, cats, year)
  }.toMap

def metaOf(id: String): (String, String, String, String) =
  metaMap.getOrElse(id, (null, null, null, null))

def paperTable(title: String, rows: Seq[(String, String)], firstColName: String): String = {
  val head =
    s"""
       |<h2>${esc(title)}</h2>
       |<table border="1" cellpadding="6" cellspacing="0">
       |<tr>
       |  <th>${esc(firstColName)}</th>
       |  <th>arXiv ID</th>
       |  <th>Title</th>
       |  <th>Authors</th>
       |  <th>Categories</th>
       |  <th>Year</th>
       |  <th>Link</th>
       |</tr>
       |""".stripMargin

  val body = rows.map { case (x, id) =>
    val (t, a, c, y) = metaOf(id)
    val url = absUrl(id)
    s"""
       |<tr>
       |  <td>${esc(x)}</td>
       |  <td><a href="${esc(url)}">${esc(id)}</a></td>
       |  <td>${esc(t)}</td>
       |  <td>${esc(a)}</td>
       |  <td>${esc(c)}</td>
       |  <td>${esc(y)}</td>
       |  <td><a href="${esc(url)}">${esc(url)}</a></td>
       |</tr>
       |""".stripMargin
  }.mkString("\n")

  head + body + "\n</table>\n"
}

def simpleTable[T1, T2](title: String, col1: String, col2: String, rows: Seq[(T1, T2)]): String = {
  val head =
    s"""
       |<h2>${esc(title)}</h2>
       |<table border="1" cellpadding="6" cellspacing="0">
       |<tr><th>${esc(col1)}</th><th>${esc(col2)}</th></tr>
       |""".stripMargin
  val body = rows.map { case (a, b) => s"<tr><td>${esc(a.toString)}</td><td>${esc(b.toString)}</td></tr>" }.mkString("\n")
  head + body + "\n</table>\n"
}

val outTableRows = topOut.map { case (deg, id) => (deg.toString, id) }.toSeq
val inTableRows  = topIn.map { case (deg, id)  => (deg.toString, id) }.toSeq
val prTableRows  = topPR.map { case (rank, id) => (f"$rank%.6f", id) }.toSeq
val triTableRows = topTriangles.map { case (cnt, id) => (cnt.toString, id) }.toSeq
val commRows     = topCommunities.toSeq.map { case (lbl, cnt) => (lbl.toString, cnt) }
val biggestCommPapersRows = papersInBiggestCommArr.toSeq.map(id => ("", id)) // first col empty

val ccRows = topCC.toSeq.map { case (n, comp) => (n, comp.toString) }
val inDegRows = inDegFirst10.toSeq.map { case (deg, cnt) => (deg, cnt) }
val bfsHistRows = hist30.toSeq.map { case (d, c) => (d, c) }

val (srcTitle, srcAuthors, srcCats, srcYear) = metaOf(sourceIdStr)
val srcUrl = absUrl(sourceIdStr)

val html =
  s"""
     |<html>
     |<head><meta charset="utf-8"><title>GraphX arXiv Results</title></head>
     |<body>
     |<h1>Citation Graph Analysis (Spark GraphX)</h1>
     |
     |<h2>Summary</h2>
     |<ul>
     |  <li>numVertices = ${numV}</li>
     |  <li>numEdges    = ${numE}</li>
     |</ul>
     |
     |${paperTable("Top 20 out-degree:", outTableRows, "OutDegree")}
     |${paperTable("Top 20 in-degree:", inTableRows, "InDegree")}
     |
     |${simpleTable("Top 10 connected components by size:", "ComponentSize", "ComponentId", ccRows)}
     |
     |${paperTable("Top 10 PageRank:", prTableRows, "PageRank")}
     |
     |<h2>Top 5 papers involved in triangles (TriangleCount):</h2>
     |${paperTable("Top 5 TriangleCount:", triTableRows, "Triangles")}
     |
     |${simpleTable("Top 10 Largest Detected Communities (Label Propagation):", "CommunityLabel", "NumPapers", commRows)}
     |
     |<h2>Examples of papers from the largest community (ID ${esc(biggestCommunityLabel.toString)}):</h2>
     |${paperTable(s"Examples (5) – Biggest Community $biggestCommunityLabel:", biggestCommPapersRows, "")}
     |
     |${simpleTable("Citation Distribution (First 10 values):", "InDegree", "NumPapers", inDegRows)}
     |<p><b>Average citations per paper (for papers cited at least once):</b> ${f"$avgInDegree%.2f"}</p>
     |
     |<h2>Degrees of Separation (BFS, undirected)</h2>
     |<p>
     |  <b>Source (most cited):</b>
     |  <a href="${esc(srcUrl)}">${esc(sourceIdStr)}</a>
     |  (inDegree=${sourceDeg})
     |</p>
     |<ul>
     |  <li>Title: ${esc(srcTitle)}</li>
     |  <li>Authors: ${esc(srcAuthors)}</li>
     |  <li>Categories: ${esc(srcCats)}</li>
     |  <li>Year: ${esc(srcYear)}</li>
     |</ul>
     |<ul>
     |  <li>Reachable nodes: ${reachableCount}</li>
     |  <li>Max hops: ${maxHops}</li>
     |  <li>Mean hops: ${f"$meanHops%.3f"}</li>
     |</ul>
     |
     |${simpleTable("Distance histogram (hops -> number of papers), first 30:", "Hops", "NumPapers", bfsHistRows)}
     |
     |<hr/>
     |<p>Link format: https://arxiv.org/abs/&lt;id&gt;</p>
     |</body>
     |</html>
     |""".stripMargin

val fs = FileSystem.get(sc.hadoopConfiguration)
val outHtmlPath = new Path(outHtmlPathStr)
if (fs.exists(outHtmlPath)) fs.delete(outHtmlPath, true)
val out = fs.create(outHtmlPath, true)
out.write(html.getBytes("UTF-8"))
out.close()

println(s"\nSaved HTML (clickable links): $outHtmlPathStr")
println("\nAnalysis completed.")

System.exit(0)
