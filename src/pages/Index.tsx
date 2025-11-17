import Header from "@/components/Header";
import Footer from "@/components/Footer";
import PostCard from "@/components/PostCard";
import Newsletter from "@/components/Newsletter";
import { getAllPosts } from "@/utils/posts";

const Index = () => {
  const posts = getAllPosts();

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        {/* Hero Section */}
        <section className="bg-gradient-to-br from-accent/5 via-primary/5 to-secondary/5 border-b border-border">
          <div className="container mx-auto px-4 py-16 md:py-24">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-accent via-primary to-secondary bg-clip-text text-transparent">
              Agentic AI for Music & Creative Arts
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-3xl">
              Exploring Letta agents, intelligent workflows, and the future of AI-assisted music production.
            </p>
          </div>
        </section>

        {/* Posts Grid */}
        <section className="container mx-auto px-4 py-12">
          <h2 className="text-3xl font-bold mb-8">Latest Posts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
            {posts.map((post) => (
              <PostCard
                key={post.slug}
                slug={post.slug}
                title={post.frontmatter.title}
                date={post.frontmatter.date}
                description={post.frontmatter.description}
                tags={post.frontmatter.tags}
              />
            ))}
          </div>

          <Newsletter />
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default Index;
