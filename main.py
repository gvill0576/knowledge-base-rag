"""
Main entry point for Knowledge Base RAG system.
"""

import sys
from src.rag import build_knowledge_base


def main():
    """Run the knowledge base system."""
    print("🚀 Knowledge Base RAG System")
    print("=" * 60)
    print("Personal knowledge base with AI-powered Q&A")
    print("=" * 60)
    
    # Parse command line arguments
    load_existing = "--load" in sys.argv
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    help_requested = "--help" in sys.argv or "-h" in sys.argv
    
    if help_requested:
        print("\nUsage:")
        print("  python main.py              # Build from scratch and run demo")
        print("  python main.py --load       # Load existing index (faster)")
        print("  python main.py --interactive # Interactive Q&A mode")
        print("  python main.py --load -i    # Load index + interactive")
        print("  python main.py --help       # Show this help")
        return
    
    # Build or load knowledge base
    kb = build_knowledge_base(load_existing=load_existing)
    
    if interactive:
        # Interactive mode
        kb.interactive()
    else:
        # Demo mode
        print(f"\n{'='*60}")
        print("DEMO MODE")
        print(f"{'='*60}")
        print("Running sample questions...")
        print("(Use --interactive for your own questions)")
        
        # Sample questions about your documents
        demo_questions = [
            "What is Python and what are its key features?",
            "What AWS services are fundamental for cloud computing?",
            "Explain continuous integration and deployment",
            "What are the different types of software tests?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n[Question {i}/{len(demo_questions)}]")
            kb.ask(question)
            
            if i < len(demo_questions):
                input("\n👉 Press Enter for next question...")
        
        print(f"\n{'='*60}")
        print("✅ Demo Complete!")
        print(f"{'='*60}")
        print("\n💡 Next steps:")
        print("   • Try interactive mode: python main.py --load --interactive")
        print("   • Add more documents to knowledge_base/")
        print("   • Modify chunk size in src/rag.py")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
