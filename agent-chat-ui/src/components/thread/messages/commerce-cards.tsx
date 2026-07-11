import { useState, type ReactNode } from "react";
import { ExternalLink, ShoppingBag, Ticket } from "lucide-react";

/** Host portion of a domain that may carry a path, e.g. "google.com/travel". */
function host(domain: string): string {
  return (domain || "").replace(/^https?:\/\//, "").split("/")[0];
}

/** Human-friendly domain: strip protocol, www, and any path. */
function prettyDomain(domain: string): string {
  return host(domain).replace(/^www\./, "");
}

/** Retailer/platform favicon with a graceful letter-avatar fallback. */
function Favicon({ domain }: { domain: string }) {
  const [failed, setFailed] = useState(false);
  const letter = (prettyDomain(domain)[0] || "?").toUpperCase();

  if (failed) {
    return (
      <div className="bg-muted text-muted-foreground flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-sm font-semibold">
        {letter}
      </div>
    );
  }
  return (
    <img
      src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(host(domain))}&sz=64`}
      alt=""
      width={36}
      height={36}
      className="bg-muted h-9 w-9 shrink-0 rounded-md object-contain p-0.5"
      onError={() => setFailed(true)}
    />
  );
}

function CardHeader({ icon, title, region }: { icon: ReactNode; title: string; region?: string }) {
  return (
    <div className="text-foreground mb-2 flex items-center gap-2 text-sm font-medium">
      {icon}
      <span>{title}</span>
      {region ? (
        <span className="text-muted-foreground text-xs font-normal">· {region}</span>
      ) : null}
    </div>
  );
}

interface Offer {
  retailer: string;
  title?: string;
  url: string;
  price?: string;
  snippet?: string;
  available?: boolean | null;
}

interface ShoppingData {
  product?: string;
  region?: string;
  currency?: string;
  offers?: Offer[];
  note?: string;
}

export function ShoppingCards({ data }: { data: ShoppingData }) {
  const offers = (data?.offers ?? []).filter((o) => o?.url);

  return (
    <div className="mx-auto w-full max-w-3xl">
      <CardHeader
        icon={<ShoppingBag className="text-muted-foreground h-4 w-4" />}
        title={data?.product ? `Where to buy ${data.product}` : "Shopping options"}
        region={data?.region}
      />
      {offers.length === 0 ? (
        <p className="border-border text-muted-foreground rounded-lg border border-dashed px-3 py-4 text-center text-xs">
          {data?.note || "No offers found, try a more specific product name."}
        </p>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          {offers.map((o, i) => (
            <a
              key={`${o.url}-${i}`}
              href={o.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group border-border bg-background hover:border-foreground/30 hover:bg-muted/40 flex items-center gap-3 rounded-lg border p-3 transition-colors"
            >
              <Favicon domain={o.retailer} />
              <div className="min-w-0 flex-1">
                <div className="text-foreground truncate text-sm font-medium">
                  {o.title || prettyDomain(o.retailer)}
                </div>
                <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
                  <span className="truncate">{prettyDomain(o.retailer)}</span>
                  {o.available === true ? (
                    <span className="shrink-0 rounded bg-green-500/15 px-1 py-0.5 text-[10px] font-medium text-green-600 dark:text-green-400">
                      In stock
                    </span>
                  ) : o.available === false ? (
                    <span className="bg-muted text-muted-foreground shrink-0 rounded px-1 py-0.5 text-[10px] font-medium">
                      Out of stock
                    </span>
                  ) : null}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {o.price ? (
                  <span className="text-foreground text-sm font-semibold tabular-nums">
                    {o.price}
                  </span>
                ) : null}
                <ExternalLink className="text-muted-foreground/60 group-hover:text-foreground h-4 w-4 transition-colors" />
              </div>
            </a>
          ))}
        </div>
      )}
      {offers.length > 0 && data?.note ? (
        <p className="text-muted-foreground/70 mt-2 text-[11px]">{data.note}</p>
      ) : null}
    </div>
  );
}

interface BookingOption {
  platform: string;
  title?: string;
  url: string;
  snippet?: string;
}

interface BookingData {
  query?: string;
  category?: string;
  region?: string;
  options?: BookingOption[];
  note?: string;
}

const CATEGORY_LABEL: Record<string, string> = {
  flight: "Flights",
  hotel: "Hotels",
  movie: "Movies",
  concert: "Concerts",
  event: "Events",
  show: "Shows",
};

export function BookingCards({ data }: { data: BookingData }) {
  const options = (data?.options ?? []).filter((o) => o?.url);
  const label = data?.category
    ? CATEGORY_LABEL[data.category] || "Booking options"
    : "Booking options";

  return (
    <div className="mx-auto w-full max-w-3xl">
      <CardHeader
        icon={<Ticket className="text-muted-foreground h-4 w-4" />}
        title={label}
        region={data?.region}
      />
      {options.length === 0 ? (
        <p className="border-border text-muted-foreground rounded-lg border border-dashed px-3 py-4 text-center text-xs">
          {data?.note || "No options found, try adding a city, date, or exact title."}
        </p>
      ) : (
        <div className="grid gap-2">
          {options.map((o, i) => (
            <a
              key={`${o.url}-${i}`}
              href={o.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group border-border bg-background hover:border-foreground/30 hover:bg-muted/40 flex items-start gap-3 rounded-lg border p-3 transition-colors"
            >
              <Favicon domain={o.platform} />
              <div className="min-w-0 flex-1">
                <div className="text-foreground truncate text-sm font-medium">
                  {o.title || prettyDomain(o.platform)}
                </div>
                {o.snippet ? (
                  <div className="text-muted-foreground line-clamp-2 text-xs">{o.snippet}</div>
                ) : null}
                <div className="text-muted-foreground/70 mt-0.5 truncate text-[11px]">
                  {prettyDomain(o.platform)}
                </div>
              </div>
              <ExternalLink className="text-muted-foreground/60 group-hover:text-foreground mt-0.5 h-4 w-4 shrink-0 transition-colors" />
            </a>
          ))}
        </div>
      )}
      {options.length > 0 && data?.note ? (
        <p className="text-muted-foreground/70 mt-2 text-[11px]">{data.note}</p>
      ) : null}
    </div>
  );
}
