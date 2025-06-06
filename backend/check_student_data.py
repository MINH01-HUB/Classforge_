from app.services.neo4j_service import neo4j_service

def check_student_data():
    print("🔍 Checking student data structure...")
    
    if not neo4j_service.connect():
        print("❌ Failed to connect to Neo4j")
        return
    
    # Check student properties
    query = """
    MATCH (s:Student)
    RETURN s LIMIT 1
    """
    result = neo4j_service.run_query(query)
    
    if result:
        student = result[0]['s']
        print("\nStudent node properties:")
        for key, value in student.items():
            print(f"- {key}: {value}")
    else:
        print("❌ No student data found")
    
    neo4j_service.close()

if __name__ == "__main__":
    check_student_data() 