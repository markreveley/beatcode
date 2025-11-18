import { useState } from "react";
import { toast } from "sonner";

const Newsletter = () => {
  const [email, setEmail] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success("Thanks for subscribing! (ConvertKit integration coming soon)");
    setEmail("");
  };

  return (
    <div className="border-2 border-foreground p-8 hatch-pattern">
      <div className="bg-background p-6 border-2 border-foreground">
        <div className="text-xs opacity-70 mb-3">ACTION: SUBSCRIBE</div>
        <h3 className="text-2xl font-bold mb-2">Newsletter</h3>
        <p className="mb-6 leading-relaxed">
          Get the latest insights on AI agents, music workflows, and creative technology delivered to your inbox.
        </p>

        {/* Workflow visualization */}
        <div className="flex items-center gap-2 text-sm mb-6 opacity-70">
          <span className="border px-2 py-1">EMAIL</span>
          <span>→</span>
          <span className="border px-2 py-1">SUBMIT</span>
          <span>→</span>
          <span className="border px-2 py-1">INBOX</span>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
          <input
            type="email"
            placeholder="your@email.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="flex-1 px-4 py-2 border-2 border-foreground bg-background text-foreground focus:outline-none focus:bg-foreground focus:text-background transition-all"
          />
          <button
            type="submit"
            className="px-6 py-2 border-2 border-foreground bg-foreground text-background hover:bg-background hover:text-foreground transition-all font-medium"
          >
            SUBSCRIBE →
          </button>
        </form>
      </div>
    </div>
  );
};

export default Newsletter;
