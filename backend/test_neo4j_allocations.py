from app.services.neo4j_service import neo4j_service

def test_allocations():
    # Connect to Neo4j
    if not neo4j_service.connect():
        print("Failed to connect to Neo4j")
        return

    # Create constraints
    neo4j_service.create_constraints()
    print("✅ Created constraints")

    # Create some test classes
    classes = [
        ("CS101", "Introduction to Programming", 30),
        ("CS102", "Data Structures", 25),
        ("CS103", "Algorithms", 20)
    ]
    
    for class_id, name, capacity in classes:
        neo4j_service.create_class(class_id, name, capacity)
    print("✅ Created test classes")

    # Create some test students
    students = [
        ("S001", "John Doe", ["CS101", "CS102"]),
        ("S002", "Jane Smith", ["CS102", "CS103"]),
        ("S003", "Bob Johnson", ["CS101", "CS103"])
    ]
    
    for student_id, name, preferences in students:
        neo4j_service.create_student(student_id, name, preferences)
    print("✅ Created test students")

    # Allocate students to classes
    allocations = [
        ("S001", "CS101"),
        ("S001", "CS102"),
        ("S002", "CS102"),
        ("S002", "CS103"),
        ("S003", "CS101")
    ]
    
    for student_id, class_id in allocations:
        neo4j_service.allocate_student_to_class(student_id, class_id)
    print("✅ Allocated students to classes")

    # Test getting students in a class
    print("\nStudents in CS101:")
    students_in_class = neo4j_service.get_students_in_class("CS101")
    for student in students_in_class:
        print(f"- {student['name']} (ID: {student['student_id']})")

    # Test getting classes for a student
    print("\nClasses for S001:")
    student_classes = neo4j_service.get_student_classes("S001")
    for class_info in student_classes:
        print(f"- {class_info['name']} (ID: {class_info['class_id']}, Capacity: {class_info['capacity']})")

    # Test getting class capacity
    print("\nClass capacities:")
    for class_id, _, _ in classes:
        capacity_info = neo4j_service.get_class_capacity(class_id)
        if capacity_info:
            info = capacity_info[0]
            print(f"- {class_id}: {info['current_enrollment']}/{info['capacity']} students")

    # Test getting all allocations
    print("\nAll allocations:")
    all_allocations = neo4j_service.get_all_allocations()
    for allocation in all_allocations:
        print(f"- {allocation['student_name']} -> {allocation['class_name']}")

    # Clean up (optional)
    # for student_id, _, _ in students:
    #     for class_id, _, _ in classes:
    #         neo4j_service.remove_student_from_class(student_id, class_id)
    # print("✅ Cleaned up allocations")

    neo4j_service.close()

if __name__ == "__main__":
    test_allocations() 