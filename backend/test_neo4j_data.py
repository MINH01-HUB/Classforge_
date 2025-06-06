from app.services.neo4j_service import neo4j_service

def check_neo4j_data():
    print("🔍 Checking Neo4j data...")
    
    # Check connection
    if not neo4j_service.connect():
        print("❌ Failed to connect to Neo4j")
        return
    
    print("✅ Connected to Neo4j")
    
    # Check nodes
    print("\n📊 Checking nodes...")
    node_query = """
    MATCH (n)
    RETURN labels(n) as type, count(*) as count
    """
    node_results = neo4j_service.run_query(node_query)
    if node_results:
        print("Node counts by type:")
        for result in node_results:
            print(f"- {result['type']}: {result['count']}")
    else:
        print("❌ No nodes found in the database")
    
    # Check relationships
    print("\n🔗 Checking relationships...")
    rel_query = """
    MATCH ()-[r]->()
    RETURN type(r) as type, count(*) as count
    """
    rel_results = neo4j_service.run_query(rel_query)
    if rel_results:
        print("Relationship counts by type:")
        for result in rel_results:
            print(f"- {result['type']}: {result['count']}")
    else:
        print("❌ No relationships found in the database")
    
    # Check specific student-class relationships
    print("\n👥 Checking student-class allocations...")
    alloc_query = """
    MATCH (s:Student)-[r:ENROLLED_IN]->(c:Class)
    RETURN s.name as student, c.name as class
    LIMIT 5
    """
    alloc_results = neo4j_service.run_query(alloc_query)
    if alloc_results:
        print("Sample student-class allocations:")
        for result in alloc_results:
            print(f"- {result['student']} → {result['class']}")
    else:
        print("❌ No student-class allocations found")
    
    neo4j_service.close()

if __name__ == "__main__":
    check_neo4j_data() 