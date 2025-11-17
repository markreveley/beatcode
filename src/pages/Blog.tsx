import { useParams } from "react-router-dom";
import { useEffect } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import MarkdownContent from "@/components/MarkdownContent";
import Newsletter from "@/components/Newsletter";
import { Badge } from "@/components/ui/badge";
import { getPostBySlug } from "@/utils/posts";
import { ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";

const Blog = () => {
  const { slug } = useParams<{ slug: string }>();
  const post = slug ? getPostBySlug(slug) : undefined;

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  if (!post) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-12">
          <h1 className="text-4xl font-bold mb-4">Post Not Found</h1>
          <p className="text-muted-foreground mb-6">The blog post you're looking for doesn't exist.</p>
          <Link to="/" className="text-primary hover:underline flex items-center gap-2">
            <ArrowLeft className="w-4 h-4" />
            Back to home
          </Link>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <article className="container mx-auto px-4 py-12 max-w-4xl">
          <Link 
            to="/" 
            className="text-muted-foreground hover:text-primary transition-colors flex items-center gap-2 mb-8"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to all posts
          </Link>
          
          <header className="mb-8">
            <time className="text-muted-foreground text-sm block mb-4">
              {new Date(post.frontmatter.date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              })}
            </time>
            <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-accent via-primary to-secondary bg-clip-text text-transparent">
              {post.frontmatter.title}
            </h1>
            <p className="text-xl text-muted-foreground mb-6">
              {post.frontmatter.description}
            </p>
            <div className="flex flex-wrap gap-2">
              {post.frontmatter.tags.map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                </Badge>
              ))}
            </div>
          </header>

          <div className="mb-12">
            <MarkdownContent content={post.content} />
          </div>

          <div className="border-t border-border pt-8">
            <Newsletter />
          </div>
        </article>
      </main>
      <Footer />
    </div>
  );
};

export default Blog;
