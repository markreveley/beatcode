import { Link } from "react-router-dom";

interface PostCardProps {
  slug: string;
  title: string;
  date: string;
  description: string;
  tags: string[];
}

const PostCard = ({ slug, title, date, description, tags }: PostCardProps) => {
  return (
    <Link to={`/blog/${slug}`} className="block group">
      <div className="border-2 border-foreground bg-background hover:bg-foreground hover:text-background transition-all p-6 h-full flex flex-col">
        {/* Date Label */}
        <div className="text-xs mb-3 opacity-70">
          <time dateTime={date}>
            {new Date(date).toLocaleDateString('en-US', {
              year: 'numeric',
              month: '2-digit',
              day: '2-digit'
            }).replace(/\//g, '-')}
          </time>
        </div>

        {/* Title */}
        <h3 className="text-lg font-bold mb-3 leading-tight">
          {title}
        </h3>

        {/* Description */}
        <p className="text-sm mb-4 opacity-80 flex-1">
          {description}
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mt-auto">
          {tags.map((tag) => (
            <span
              key={tag}
              className="text-xs border border-current px-2 py-1"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Arrow indicator */}
        <div className="mt-4 pt-4 border-t border-current/20 text-right text-sm">
          READ →
        </div>
      </div>
    </Link>
  );
};

export default PostCard;
