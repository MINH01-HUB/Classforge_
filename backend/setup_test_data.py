from app.services.neo4j_service import neo4j_service

def setup_test_data():
    print("🔄 Setting up test data...")
    
    if not neo4j_service.connect():
        print("❌ Failed to connect to Neo4j")
        return
    
    # Create test classes
    classes = [
        {"id": "CS101", "name": "Introduction to Programming", "capacity": 30},
        {"id": "CS102", "name": "Data Structures", "capacity": 25},
        {"id": "CS103", "name": "Algorithms", "capacity": 25},
        {"id": "CS104", "name": "Database Systems", "capacity": 20},
        {"id": "CS105", "name": "Web Development", "capacity": 20}
    ]
    
    print("\n📚 Creating classes...")
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
    
    # Get all students
    print("\n👥 Getting students...")
    student_query = "MATCH (s:Student) RETURN s.student_id as id, s.name as name"
    students = neo4j_service.run_query(student_query)
    
    if not students:
        print("❌ No students found in the database")
        return
    
    print(f"Found {len(students)} students")
    
    # Assign students to classes
    print("\n📝 Assigning students to classes...")
    for i, student in enumerate(students):
        # Assign each student to 2-3 random classes
        class_ids = [classes[j % len(classes)]["id"] for j in range(i, i + 2 + (i % 2))]
        
        for class_id in class_ids:
            query = """
            MATCH (s:Student {student_id: $student_id})
            MATCH (c:Class {class_id: $class_id})
            CREATE (s)-[:ENROLLED_IN]->(c)
            """
            neo4j_service.run_query(query, {
                "student_id": student["id"],
                "class_id": class_id
            })
    
    print("✅ Test data setup complete!")
    neo4j_service.close()

if __name__ == "__main__":
    setup_test_data() 