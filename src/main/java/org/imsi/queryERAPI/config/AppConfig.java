package org.imsi.queryERAPI.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

/**
 * Configuration to load application properties
 */
@Configuration
@PropertySource("classpath:config.properties")
public class AppConfig {
    // This class enables loading of config.properties
    // Properties can be injected using @Value annotation
}
