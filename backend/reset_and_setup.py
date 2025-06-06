from app.services.neo4j_service import neo4j_service

def reset_and_setup():
    print("🔄 Resetting and setting up fresh test data...")
    
    if not neo4j_service.connect():
        print("❌ Failed to connect to Neo4j")
        return
    
    # Clear all data
    print("\n🧹 Clearing existing data...")
    neo4j_service.run_query("MATCH (n) DETACH DELETE n")
    
    # Create test students
    print("\n👥 Creating test students...")
    students = [
        {"id": "S001", "name": "John Doe"},
        {"id": "S002", "name": "Jane Smith"},
        {"id": "S003", "name": "Bob Johnson"},
        {"id": "S004", "name": "Alice Brown"},
        {"id": "S005", "name": "Charlie Davis"}
    ]
    
    for student in students:
        query = """
        CREATE (s:Student {
            student_id: $id,
            name: $name
        })
        """
        neo4j_service.run_query(query, student)
        print(f"✅ Created student: {student['name']}")
    
    # Create test classes
    print("\n📚 Creating test classes...")
    classes = [
        {"id": "CS101", "name": "Introduction to Programming", "capacity": 30},
        {"id": "CS102", "name": "Data Structures", "capacity": 25},
        {"id": "CS103", "name": "Algorithms", "capacity": 25}
    ]
    
    for class_data in classes:
        query = """
        CREATE (c:Class {
            class_id: $id,
            name: $name,
            capacity: $capacity
        })
        """
        neo4j_service.run_query(query, class_data)
        print(f"✅ Created class: {class_data['name']}")
    
    # Create enrollments
    print("\n📝 Creating enrollments...")
    enrollments = [
        ("S001", "CS101"),
        ("S001", "CS102"),
        ("S002", "CS101"),
        ("S002", "CS103"),
        ("S003", "CS101"),
        ("S003", "CS102"),
        ("S004", "CS102"),
        ("S004", "CS103"),
        ("S005", "CS101"),
        ("S005", "CS103")
    ]
    
    for student_id, class_id in enrollments:
        query = """
        MATCH (s:Student {student_id: $student_id})
        MATCH (c:Class {class_id: $class_id})
        CREATE (s)-[:ENROLLED_IN]->(c)
        """
        neo4j_service.run_query(query, {
            "student_id": student_id,
            "class_id": class_id
        })
        print(f"✅ Enrolled student {student_id} in class {class_id}")
    
    print("\n✅ Test data setup complete!")
    neo4j_service.close()

if __name__ == "__main__":
    reset_and_setup() 